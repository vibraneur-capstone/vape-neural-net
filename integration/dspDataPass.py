def dspDataParse(dspData):
    dspData = dspData[2:-1].split(',')
    array = []
    for data in dspData:
        data = data.split(':')
        array.append(data)
    inputA = [[float(array[4][1])], [float(array[5][1])],[float(array[6][1])],[float(array[7][1])]]
    processedDspData = dict({
        'id':array[0][1],
        'sensorDataPointId':array[1][1][1:-1],
        'sensorId': array[2][1][1:-1],
        'timestamp': array[3][1][1:]+ ':' + array[3][2] + ':' + array[3][3][:-1],
        'inputA': inputA
    })
    return processedDspData
    
