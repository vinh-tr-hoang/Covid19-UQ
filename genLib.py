import csv
def report(filename, **kwargs):
    with open(filename,'w',newline='') as file:
        writer = csv.writer(file, dialect='excel-tab')
        writer.writerows([[key for key in kwargs.keys()],[kwargs[key] for key in kwargs.keys()]])

