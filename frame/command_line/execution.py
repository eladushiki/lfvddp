import os
from subprocess import PIPE, Popen

def execute_in_process(command):
    process = Popen(
        args=command,
        stdout=PIPE,
        stderr=PIPE,
    shell=True)
    out,err=process.communicate()
    returncode=process.returncode
    return str(out),str(err),returncode
