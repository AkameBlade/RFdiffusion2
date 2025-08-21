import subprocess
import re
import os
import logging

logger = logging.getLogger(__name__)


def slurm_submit(cmd, p='cpu', c=1, mem=2, gres=None, J=None, wait_for=[], hold_until_finished=False, log=False, **kwargs):
    '''
    wait_for = wait for these slurm jobids to exist okay
    hold_until_finished =  if True, don't return command line control until the slurm job is done
    '''
    job_name = J if J else os.environ["USER"]+'_auto_submit'
    log_file = f'%A_%a_{J}.log' if log else '/dev/null'
    # The RAM on this gpu is busted, raises ECC errors on torch.load
    exclude_gpu = "--exclude=gpu135,gpu111,gpu51,gpu53,gpu21,gpu130,gpu76,gpu83,gpu72" if gres and gres.startswith('gpu') else "--exclude=dig193,dig182"

    # Inexplicably, the interfaceAF2predict_bcov.py script does not play
    # nicely with the 4000Ada Remote GPUs.
    is_interface_af2_job = 'jobs.score.af2_initial_guess' in cmd
    ada4000remote_nodes = 'gpu[142-157]'
    if is_interface_af2_job:
        exclude_gpu += f",{ada4000remote_nodes}"

    cmd_sbatch = f'sbatch --wrap "{cmd}" -p {p} -c {c} --mem {mem}g '\
        f'-J {job_name} '\
        f'{f"--gres {gres}" if gres else ""} '\
        f'{exclude_gpu} '\
        f'{"-W" if hold_until_finished else ""} '\
        f'{"--dependency afterok:" + ":".join(map(str, wait_for)) if wait_for else ""} '\
        f'-o {log_file} ' \
        f'--export PYTHONPATH={os.environ["PYTHONPATH"]} '
    cmd_sbatch += ' '.join([f'{"--"+k if len(k)>1 else "-"+k} {v}' for k,v in kwargs.items() if v is not None])
    print(f'cmd_sbatch: {cmd_sbatch}')

    proc = subprocess.run(cmd_sbatch, shell=True, stdout=subprocess.PIPE)
    slurm_job = re.findall(r'\d+', str(proc.stdout))[0]
    slurm_job = int(slurm_job)

    return slurm_job, proc

def line_count(path):
    with open(path) as fh:
        return len(fh.readlines())

def array_submit(job_list_file, p='gpu', gres='gpu:rtx2080:1', wait_for=None, log=False, in_proc=False, already_ran=None, mem=12, **kwargs):
    logger.info(f'array_submit: in_proc: {in_proc}')

    if already_ran is not None:
        job_list_file = prune_jobs(job_list_file, already_ran)

    job_count = line_count(job_list_file)
    logger.info(job_count)
    if job_count == 0:
        return -1, None

    if in_proc:
        with open(job_list_file) as f:
            jobs = f.readlines()
        for job in jobs:
            # For logging (hides retcode)
            # job = re.sub('>>', '2>&1 | tee', job)
            job = re.sub('>>.*', '', job)

            logger.info(f'running job after: {job}')

            proc = subprocess.run(job, shell=True, stderr=subprocess.PIPE)
            if proc.returncode != 0:
                raise Exception(f'FAILED command: {job}. \n'
                                f'stderr: {proc.stderr.decode()}')
        return -1, None
    return slurm_submit(
        cmd = 'eval \\`sed -n \\${SLURM_ARRAY_TASK_ID}p '+job_list_file+'\\`',
        a = f'1-$(cat {job_list_file} | wc -l)',
        p = p,
        c = 1,
        mem = mem,
        gres = gres,
        wait_for = wait_for,
        log = log,
        **kwargs
    )

def prune_jobs(
    job_list_file,
    already_ran):
    '''
    Inputs:
        job_list_file: path to file containing newline delimited jobs
        already_ran: dictionary mapping each line in job_list_file
            to a function which returns True if the expected outputs already exist.
    Returns:
        Path to pruned job list.
    '''
    job_list_file_pruned = job_list_file + '.pruned'
    with open(job_list_file) as f:
        jobs = f.readlines()

    pruned_jobs = []
    for job in jobs:
        job = job.strip()
        logger.info(f'Checking {job=}')
        if not already_ran[job]():
            pruned_jobs.append(job)
    with open(job_list_file_pruned, 'w') as f:
        f.writelines(j + '\n' for j in pruned_jobs)
    
    logger.info(f"Pruned job_list_file: {job_list_file} from {len(jobs)} to {len(pruned_jobs)} jobs")
    return job_list_file_pruned

