import xlwt
from xlwt import Workbook

event_acc = []
evtype_accs = [] 
argument_accs = [] 
argtype_accs = [] 
role_accs = []
overall_accs = []

with open("Logging/Results_logger.txt", 'r') as f:
    file = f.readlines()

    i = 0

    for line in file:
        acc = line.split(':')[1].strip()
        if i%6 == 0:
            event_acc.append(acc)
        if i%6 == 1:
            evtype_accs.append(acc)
        if i%6 == 2:
            argument_accs.append(acc)
        if i%6 == 3:
            argtype_accs.append(acc)
        if i%6 == 4:
            role_accs.append(acc)
        if i%6 == 5:
            overall_accs.append(acc)
        i += 1

# write to csv

wb = Workbook()

sheet1 = wb.add_sheet('Sheet 1')

sheet1.write(0, 1, 'Trigger Word Accuracy')
sheet1.write(0, 2, 'Event Type Accuracy')
sheet1.write(0, 3, 'Argument Identification Accuracy')
sheet1.write(0, 4, 'Argument Type Accuracy')
sheet1.write(0, 5, 'Argument Role Accuracy')
sheet1.write(0, 6, 'Overall F Score')

for i in range(len(event_acc)):
    sheet1.write(i+1, 1, event_acc[i] )
for i in range(len(evtype_accs)):
    sheet1.write(i+1, 2, evtype_accs[i] )
for i in range(len(argument_accs)):
    sheet1.write(i+1, 3, argument_accs[i] )
for i in range(len(argtype_accs)):
    sheet1.write(i+1, 4, argtype_accs[i] )
for i in range(len(role_accs)):
    sheet1.write(i+1, 5, role_accs[i] )
for i in range(len(overall_accs)):
    sheet1.write(i+1, 6, overall_accs[i] )

wb.save('Logging/DataLogging.xls')