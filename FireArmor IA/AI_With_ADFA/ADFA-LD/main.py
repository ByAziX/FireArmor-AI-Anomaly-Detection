import csv
#replace syscall name with syscall number whith the file label.csv in the file syscall_names.txt and write the result in syscall_num.txt

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
                # si syscall_name commence par un _ alors on le supprime
                
                    
                syscall_number = syscall_dict_sys[syscall_name]
                syscall_number = syscall_number.strip()
                output_file.write(syscall_number + ' ')

            elif syscall_name in syscall_dict_NR:
                syscall_number = syscall_dict_NR[syscall_name]
                syscall_number = syscall_number.strip()
                output_file.write(syscall_number + ' ')

            else:
                print('Syscall name not found: ' + syscall_name)
        
        



replace_syscall_with_number('FireArmor IA/AI_With_ADFA/ADFA-LD/syscall_names.txt', 'FireArmor IA/AI_With_ADFA/ADFA-LD/label.csv', 'FireArmor IA/AI_With_ADFA/ADFA-LD/syscall_num.txt')
