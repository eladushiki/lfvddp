from frame.command_line.execution import execute_in_process



def submit_save_jobs(fsubname,N_jobs,walltime="06:00:00",io=5,mem=None,cores=None,waitjobids=[],mail=False):
    ## Get submit command
    user = execute_in_process('whoami')[0][2:-3]
    host = execute_in_process('hostname')[0][2:-3]
    # email = f'{user}@{host}'
    subcmd="qsub -l walltime=%s,io=%s"%(walltime,io)
    waitjobids = execute_in_process(f"qstat -u {user} | tail -n {N_jobs} | sed -e 's/\..*$//' | tr '\n' ' '")[0][2:-1].split()
    if mem!=None:
        subcmd+=",mem=%sg"%mem # Default in farm is 2g
    if cores!=None:
        subcmd+=",ppn=%s"%cores 
    # if mail:
    #     subcmd+=f" -M {email}"
    if waitjobids!=[]:
        subcmd+=" -W depend=afterok"
        for wjid in waitjobids:
            subcmd+=":%s.wipp-pbs"%wjid
    subcmd+=" %s"%fsubname
    ## Submit
    returncode=""
    tries=[]
    while returncode!=0 and len(tries)<50:
        if returncode!="":
            print(returncode,err)
        out,err,returncode=execute_in_process(subcmd)
        if returncode==228:
            # warn("Too many jobs submitted")
            return None
        tries.append(returncode)
    if returncode!=0:
        # warn("Problem submitting job",fname)
        print("Submit command:",subcmd)
        print("Returncodes per try:")
        print(tries)
        return None
    ## Return jobID
    jobID=out.split('.')[0].rstrip()
    print(jobID,fsubname)
    return jobID


def submit_job(fsubname,walltime="06:00:00",io=5,mem=None,cores=None,waitjobids=[]):
    ## Get submit command
    subcmd="qsub -l walltime=%s,io=%s"%(walltime,io)
    if mem!=None:
        subcmd+=",mem=%sg"%mem # Default in farm is 2g
    if cores!=None:
        subcmd+=",ppn=%s"%cores
    if waitjobids!=[]:
        subcmd+=" -W depend=afterany"
        for wjid in waitjobids:
            subcmd+=":%s.wipp-pbs"%wjid
    subcmd+=" %s"%fsubname
    ## Submit
    returncode=""
    tries=[]
    while returncode!=0 and len(tries)<50:
        if returncode!="":
            print(returncode,err)
        out,err,returncode=execute_in_process(subcmd)
        if returncode==228:
            # warn("Too many jobs submitted")
            return None
        tries.append(returncode)
    if returncode!=0:
        # warn("Problem submitting job",fname)
        print("Submit command:",subcmd)
        print("Returncodes per try:")
        print(tries)
        return None
    ## Return jobID
    jobID=out.split('.')[0].rstrip()
    print(jobID,fsubname)
    return jobID


def prepare_submit_file(fsubname,setupLines,cmdLines,setupATLAS=True,queue="N",shortname=""):
    jobname=shortname if shortname else fsubname.rsplit('/',1)[1].split('.')[0]
    flogname=fsubname.replace('.sh','.log')
    fsub=open(fsubname,"w")
    lines=[
        "#!/bin/zsh",
        "",
        "#PBS -j oe",
        "#PBS -m n",
        "#PBS -o %s"%flogname,
        "#PBS -q %s"%queue,
        "#PBS -N %s"%jobname,
        "",
        "echo \"Starting on `hostname`, `date`\"",
        "echo \"jobs id: ${PBS_JOBID}\"",
        ""]
    if setupATLAS:
        lines+=[
            "export ATLAS_LOCAL_ROOT_BASE=/cvmfs/atlas.cern.ch/repo/ATLASLocalRootBase",
            "source ${ATLAS_LOCAL_ROOT_BASE}/user/atlasLocalSetup.sh",""]
    lines+=setupLines
    lines+=["","#-------------------------------------------------------------------#"]
    lines+=cmdLines
    lines+=["#-------------------------------------------------------------------#",""]
    lines+=["echo \"Done, `date`\""]
    for l in lines:
        fsub.write(l+"\n")
    fsub.close()
