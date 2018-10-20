import networkx as nx
import os

file_read = open('send_t_5.txt', 'r')
file_read_2 = open('receive_t_5.txt', 'r')
file_write = open("new_enron_email.txt", 'r')

lines_read = file_read.readlines()
lines_read_2 = file_read_2.readlines()
last_line = file_write.readlines()[-1]
last_time = int(last_line.split()[2])
t = last_time
s = ''
file_write = open("new_enron_email.txt", 'a+')

# for line in lines_read:
#     count += 1
#     line_read = line.split()
#     if s != line_read[2]:
#         t += 1
#         s = line_read[2]
#     line_write = line_read[0] + ' ' + line_read[1] + ' ' + str(t)
#     if count != length:
#         file_write.writelines(line_write + '\n')
#     else:
#         file_write.writelines(line_write)
#
# file_read.close()
# file_write.close()
#
for line in lines_read:
    for line_2 in lines_read_2:
        line_read = line.split()
        line_read_2 = line_2.split()
        if line_read[2] != line_read_2[2]:
            if int(line_read[2]) > int(line_read_2[2]):
                continue
            else:
                break
        if line_read[0] == line_read_2[0]:
            if s != line_read[2]:
                t += 1
                s = line_read[2]
            line_write = line_read[1] + ' ' + line_read_2[1] + ' ' + str(t)
            file_write.writelines('\n' + line_write)


file_read.close()
file_write.close()