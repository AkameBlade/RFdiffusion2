#!/usr/bin/env -S /bin/sh -c '"$(dirname "$0")/../exec/rf_diffusion_aa_shebang.sh" "$0" "$@"'
#
# Takes a folder of pdb & trb files, generates MPNN features (fixing AAs at
# contig positions), makes list of MPNN jobs on batches of those designs,
# and optionally submits slurm array job and outputs job ID
# 
from collections import defaultdict
import sys
import os
import glob
import numpy as np
import tqdm
import copy
import logging
import pickle
import hydra
from hydra.core.hydra_config import HydraConfig
from rf_diffusion.benchmark.util import slurm_tools

from paths import evaluate_path

import rf_diffusion
PKG_DIR = rf_diffusion.__path__[0]
REPO_DIR = os.path.dirname(PKG_DIR)
mpnn_script = os.path.join(REPO_DIR, 'fused_mpnn/run.py')
logger = logging.getLogger(__name__)


def memoize_to_disk(file_name):
    file_name = file_name + '.memo'
    cache_key_hash_path = file_name + '.hash'
    def decorator(func):
        def new_func(*args, cache_key=None, **kwargs):
            cache_valid = os.path.exists(file_name) and os.path.exists(cache_key_hash_path)
            if cache_valid:
                with open(cache_key_hash_path, 'rb') as fh:
                    got_cache_key = pickle.load(fh)
                cache_valid = got_cache_key == cache_key
            hit_or_miss = 'hit' if cache_valid else 'miss'
            logger.debug(f'mpnn_designs_v2.memoize_to_disk: cache {hit_or_miss}')
            if cache_valid:
                with open(file_name, 'rb') as fh:
                    return pickle.load(fh)
            o = func(*args, **kwargs)
            with open(file_name, 'wb') as fh:
                pickle.dump(o, fh)
            with open(cache_key_hash_path, 'wb') as fh:
                pickle.dump(cache_key, fh)
            return o
        return new_func
    return decorator

@hydra.main(version_base=None, config_path='configs/', config_name='mpnn_designs')
def main(conf: HydraConfig) -> list[int]:
    '''
    ### Expected conf keys ###
    datadir:                Folder of designs to score.
    chunk:                  How many designs to process in each job.
    num_seq_per_target:     How many mpnn sequences per design.
    use_ligand:             Use ligandMPNN.
    cautious:               Skip design if output file exists.

    slurm:
        J:          Job name
        p:          Partition
        gres:       Gres specification
        submit:     False = Do not submit slurm array job, only generate job list. <True, False>
        in_proc:    Run slurm array job on the current node? <True, False>
        keep_logs:  Keep the slurm logs? <True, False>
    '''
    filenames = glob.glob(conf.datadir+'/*.pdb')
    # Filter out those missing TRBs
    filenames = [fn for fn in filenames if os.path.exists(os.path.splitext(fn)[0] + '.trb')]

    logger.info(f'mpnn_designs_v2.main: {len(filenames)} backbones received for sequence fitting')
    
    if not conf.use_ligand:
        return run_mpnn(conf, filenames)


    @memoize_to_disk(os.path.join(conf.datadir, 'filenames_by_ligand_presence'))
    def categorize_by_ligand_presence():
        '''
        Categorizes a list of PDBs by presence/absence of any ligand.  Caches to disk.
        '''
        filenames_by_ligand_presence = defaultdict(list)
        logger.info(f'Categorizing {len(filenames)} PDBs by presence/absence of ligand')
        for fn in tqdm.tqdm(filenames):
            trb_path = os.path.splitext(fn)[0] + '.trb'
            trb = np.load(trb_path, allow_pickle=True)
            has_ligand = bool(trb['config']['inference']['ligand'])
            filenames_by_ligand_presence[has_ligand].append(fn)
        return filenames_by_ligand_presence
        
    if conf.ligand_present_for_all:
        filenames_by_ligand_presence = {True: filenames}
    else:
        filenames_by_ligand_presence = categorize_by_ligand_presence(cache_key=filenames)
    job_ids = []
    for use_ligand, filenames in filenames_by_ligand_presence.items():
        conf_for_mpnn_flavor = copy.deepcopy(conf)
        conf_for_mpnn_flavor.use_ligand = use_ligand
        job_ids.extend(run_mpnn(conf_for_mpnn_flavor, filenames))
    return job_ids


def get_binary(in_proc):
    sif_path = evaluate_path('REPO_ROOT/rf_diffusion/exec/mlfold.sif')
    return f'/usr/bin/apptainer exec --nv {sif_path} python -s -u'

def run_mpnn(conf, filenames):
    '''
    Takes a folder of pdb & trb files, generates MPNN features (fixing AAs at
    contig positions), makes list of MPNN jobs on batches of those designs,
    and optionally submits slurm array job and outputs job ID.
    '''

    model_type = 'protein_mpnn'
    mpnn_flavor = 'mpnn'
    if conf.use_ligand:
        mpnn_flavor = 'ligmpnn'
        model_type = 'ligand_mpnn'

    mpnn_folder = conf.datadir+f'/{mpnn_flavor}/'
    os.makedirs(mpnn_folder, exist_ok=True)

    logger.info(f'mpnn_designs_v2.run_mpnn: {len(filenames)} backbones received for sequence fitting')

    # skip designs that have already been done
    if conf.cautious:
        filtered = [fn for fn in filenames 
            if not os.path.exists(mpnn_folder+'/seqs/'+os.path.basename(fn).replace('.pdb','.fa'))]
        
        completed = set(filenames).difference(filtered)
        logger.info(f'{len(completed)}/{len(filtered)} already complete, skipping these')

        if conf.unsafe_skip_parsing and len(completed):
            raise Exception('do not combine unsafe_skip_parsing with cautious')
        filenames = filtered

    if conf.chunk == -1:
        conf.chunk = len(filenames)

    prev_job = None
    if not conf.unsafe_skip_parsing:
        # run parser script
        job_fn = conf.datadir + f'/jobs.{mpnn_flavor}.parse.list'
        job_list_file = open(job_fn, 'w') if conf.slurm.submit else sys.stdout
        parse_script = f'{PKG_DIR}/benchmark/util/parse_multiple_chains_v2.py'
        for i in range(0, len(filenames), conf.chunk):
            with open(mpnn_folder+f'parse_multiple_chains.list.{i}','w') as outf:
                for fn in filenames[i:i+conf.chunk]:
                    print(fn,file=outf)
            print(f'{parse_script} --input_files {mpnn_folder}/parse_multiple_chains.list.{i} '\
                f'--datadir {conf.datadir} '\
                f'--output_parsed {mpnn_folder}/pdbs_{i}.jsonl '\
                f'--output_fixed_pos {mpnn_folder}/pdbs_position_fixed_{i}.jsonl', file=job_list_file)
        if conf.slurm.submit: job_list_file.close()
        logger.info(f'Creating {int(np.ceil(len(filenames)/conf.chunk))} jobs to preprocess {len(filenames)} designs for MPNN')

        # submit to slurm
        job_ids = []
        if conf.slurm.submit:
            job_name = ('ligmpnn_pre' if conf.use_ligand else 'mpnn_pre') + conf.preprocessing_slurm.J
            job_id, proc = slurm_tools.array_submit(job_fn, p = conf.preprocessing_slurm.p, gres=conf.preprocessing_slurm.gres, log=conf.preprocessing_slurm.keep_logs, J=job_name, in_proc=conf.preprocessing_slurm.in_proc)
            if job_id > 0:
                job_ids.append(job_id)
            logger.info(f'Submitted array job {job_id} with {int(np.ceil(len(filenames)/conf.chunk))} jobs to preprocess {len(filenames)} designs for MPNN')

            prev_job = job_id

    job_fn = conf.datadir + f'/jobs.{mpnn_flavor}.list'
    job_list_file = open(job_fn, 'w') if conf.slurm.submit else sys.stdout

    for i in range(0, len(filenames), conf.chunk):

        print(f'{get_binary(conf.slurm.in_proc)} {mpnn_script} '\
            f'--pdb_path_multi {mpnn_folder}pdbs_position_fixed_{i}.jsonl '\
            f'--fixed_residues_multi {mpnn_folder}pdbs_position_fixed_{i}.jsonl '\
            f'--model_type {model_type} '\
            f'--pack_side_chains 1 '\
            f'--out_folder {mpnn_folder} '\
            f'--temperature="0.1" '\
            f'--batch_size {conf.num_seq_per_target} '\
            f'--ligand_mpnn_use_side_chain_context 1 '\
            f'--zero_indexed 1 '\
            f'--packed_suffix "" '\
            f'--omit_AA {conf.omit_AA} ',
            f'--repack_everything {1 if conf.pack_motif else 0}',
            '--force_hetatm 1',
            file=job_list_file)
    if conf.slurm.submit: job_list_file.close()

    # submit job
    if conf.slurm.submit:
        if conf.slurm.J is not None:
            job_name = conf.slurm.J
        else:
            job_name = 'mpnn_' + os.path.basename(conf.datadir.strip('/'))
        pre = 'ligand_' if conf.use_ligand else 'protein_'
        job_name = pre + job_name
        job_id, proc = slurm_tools.array_submit(job_fn, p = conf.slurm.p, gres=conf.slurm.gres, log=conf.slurm.keep_logs, J=job_name, wait_for=[prev_job], in_proc=conf.slurm.in_proc)
        if job_id > 0:
            job_ids.append(job_id)
        logger.info(f'Submitted array job {job_id} with {int(np.ceil(len(filenames)/conf.chunk))} jobs to MPNN {len(filenames)} designs')

    return job_ids

if __name__ == "__main__":
    main()
