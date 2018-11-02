import linecache
import csv


def txt_to_csv(file_path: str, start: int = 1, end: int = 1):
    csv_file_path = file_path.rstrip('.txt') + '.csv'
    print(csv_file_path)
    out_file = open(csv_file_path, mode='w', newline='')
    writer = csv.writer(out_file)
    writer.writerow(['node1', 'node2', 'timestamp'])
    print("start writing {0}...".format(csv_file_path))
    for i in range(start, end + 1):
        temp_line_str = linecache.getline(file_path, i)
        temp_line = temp_line_str.strip().strip('\n').split(' ')
        temp_node1 = int(temp_line[0])
        temp_node2 = int(temp_line[1])
        timestamp = int(temp_line[2])
        line_to_write = [temp_node1, temp_node2, timestamp]
        writer.writerow(line_to_write)
    print('done writing.')


if __name__ == '__main__':
    txt_to_csv(r"../Datasets/CollegeMsg temporal dataset/new_CollegeMsg.txt", 1, 59835)
