import csv

# deduplicated/data-cleaned/etc version in https://docs.google.com/spreadsheets/d/1DPtMgi-TpM2pVC2_JvROYSqgHKEWks1TTsh-3REW8Ys/edit?usp=sharing

def reformat_to_csv(input_file, output_file):
    with open(input_file, 'r') as f_in, open(output_file, 'w', newline='') as f_out:
        reader = f_in.readlines()
        writer = csv.writer(f_out)
        writer.writerow(['File', 'Module', 'Test'])  # Write header

        current_file = ''
        current_module = ''

        for line in reader:
            stripped_line = line.strip()
            if stripped_line[:15] == "Listing tests from"[:15]:  # It's a file
                current_file = stripped_line.split(' ')[-1][12:]
            elif stripped_line[-1] == '.':  # It's a module, remove the build_bin
                current_module = stripped_line[:-1]
            else:  # It's a test
                writer.writerow([current_file, current_module, stripped_line])

reformat_to_csv('test_list_original.txt', 'test_list.csv')