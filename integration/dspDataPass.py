class dspDataParse:
    def __init__(self, raw_data):
        self.data = raw_data

    def get_neural_nets_input(dspData):
        dspData = str(dspData)[3:-2]
        dspDataA = dspData.split(',')
        dspDataA = get_input_A(dspDataA)
        dspDataB = dspData.split('"fft":')
        dspDataB = get_input_B(dspDataB)
        processedDspData = dict({
            'id':array[0][1],
            'sensorDataPointId':array[1][1][1:-1],
            'sensorId': array[2][1][1:-1],
            'timestamp': array[3][1][1:]+ ':' + array[3][2] + ':' + array[3][3][:-1],
            'inputA': inputA,
            'inputB': inputB
        })
        return processedDspData

    def get_input_A(dspDataA):
        array = []
        for data in dspDataA:
            data = data.split(':')
            array.append(data)
        inputA = [[array[4][1]], [array[5][1]],[array[6][1]],[array[7][1]]]
        return map(lambda x:float(x), inputA)

    def get_input_B(dspDataB):
        dspDataB = dspDataB[1][1:-1]
        dspDataB = dspDataB.split(',') 
        return map(lambda x:float(x), dspDataB)

    def is_valid(data):
        return True
    

 
    
#