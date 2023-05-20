import csv
#replace syscall name with syscall number whith the file label.csv in the file syscall_names.txt and write the result in syscall_num.txt

def replace_syscall_with_number(input_file, csv_file, output_file):
    syscall_dict = {}
    with open(csv_file, 'r') as csv_file:
        reader = csv.reader(csv_file)
        for row in reader:
            syscall_dict[row[0]] = (row[1])
            print(syscall_dict)

    with open(input_file, 'r') as input_file, open(output_file, 'w') as output_file:
        for line in input_file:
            syscall_name = line.strip()
            if syscall_name in syscall_dict:
                syscall_number = syscall_dict[syscall_name]
                output_file.write(f"{syscall_number}\n")
            else:
                output_file.write(f"{syscall_name}\n")


replace_syscall_with_number('syscall_names.txt', 'label.csv', 'syscall_num.txt')
