import csv
#replace syscall name with syscall number whith the file label.csv in the file syscall_names.txt and write the result in syscall_num.txt

def replace_syscall_with_number(input_file, csv_file, output_file):
    syscall_dict = {}
    with open(csv_file, 'r') as csv_file:
        reader = csv.reader(csv_file)
        for row in reader:
            if not row or len(row) < 2:  # Skip the row if it is empty or if it doesn't have at least 2 elements
                continue
            syscall_name = row[2] 
            # __NR_ftruncate get the syscall name without __NR_
            syscall_name = syscall_name.replace('sys_', '').strip()
            syscall_number = row[1] 
            syscall_dict[syscall_name] = syscall_number
    

    with open(input_file, 'r') as input_file, open(output_file, 'w') as output_file:
        for line in input_file:
            syscall_name = line.strip()
            if syscall_name in syscall_dict:
                syscall_number = syscall_dict[syscall_name]
                output_file.write(syscall_number + '\n')
            else:
                print('Syscall name not found: ' + syscall_name)
        
        



replace_syscall_with_number('FireArmor IA/AI_With_ADFA/ADFA-LD/syscall_names.txt', 'FireArmor IA/AI_With_ADFA/ADFA-LD/label.csv', 'FireArmor IA/AI_With_ADFA/ADFA-LD/syscall_num.txt')
