from os import listdir
from os.path import isfile, join
import datetime

# Set path where data is located and target location of new data
path = './dataset/bearing/NASA_bearing_dataset/1st_test/'
target = './groundtruth/'
outputname = 'gt1.dat'

# For every file in the path, check if it is a file and add it to our list of files
data = [f for f in listdir(path) if isfile(join(path, f))]

def makeDate(date):
    return datetime.datetime(date[0], date[1], date[2], date[3], date[4], date[5])

# Find the start and end dates, split them, map them to integers and then convert them to datetime objects
startDate = makeDate(list(map(int,data[0].split('.'))))
endDate = makeDate(list(map(int,data[len(data)-1].split('.'))))

# Find the total duration in seconds
duration = (endDate - startDate).total_seconds()
print(duration)

# For every file in our list of files, do this loop
for file in data:
    
    # Open up our file
    input = open(path + file,"r")
    
    # Get the date from the list of files 
    date = makeDate(list(map(int,file.split('.'))))
    
    # Find the difference and find the percent difference in seconds
    difference = (endDate - date).total_seconds()
    percentDiff = (difference/duration)*100
    
    print("%i:%i:%i %i/%i/%i - %i seconds to end - %f %%" % (date.hour, date.minute, date.second, date.year, date.month, date.day, difference, percentDiff))
    
    # Create file and write the difference ratio to it
    output = open(target + outputname, "a+")
    output.write("%f\n" % (percentDiff/100))
    
input.close()
#output.close()