from datagenerering import lag_tre_multi

cases = ['TONSTAD_TWO_Q20_TWO2',
'TONSTAD_TWO_Q20_TWO3',
'TONSTAD_TWO_Q20_TWO',
'TONSTAD_TWO_Q40_TWO',
'TONSTAD_TWO_Q60_TWO',
'TONSTAD_TWO_Q80_TWO',
'TONSTAD_TWO_Q100_TWO',
'TONSTAD_TWO_Q120_TWO',
'TONSTAD_TWO_Q140_TWO',
'Tonstad_THREE_Q20_THREE',
'Tonstad_THREE_Q40_THREE',
'Tonstad_THREE_Q40_THREE_EXTRA',
'Tonstad_THREE_Q40_THREE FINAL',
'Tonstad_THREE_Q60_THREE',
'Tonstad_THREE_Q80EXTRA2_THREE',
'Tonstad_THREE_Q80_THREE',
'Tonstad_THREE_Q80_THREE_EXTRA',
'Tonstad_THREE_Q100_EXTRA2_THREE',
'Tonstad_THREE_Q100_THREE',
'Tonstad_THREE_Q100_THREE_EXTRA3',
'Tonstad_THREE_Q100_THREE_EXTRA',
'TONSTAD_FOUR_Q20_FOUR CHECK',
'TONSTAD_FOUR_Q20_FOUR REPEAT',
'TONSTAD_FOUR_Q20_FOUR TRIALONE',
'TONSTAD_FOUR_Q60_FOUR REPEAT',
'TONSTAD_FOUR_Q100_FOUR DT',
'TONSTAD_FOUR_Q40_FOUR',
'TONSTAD_FOUR_Q40_REPEAT',
'TONSTAD_FOUR_Q60_FOUR',
'TONSTAD_FOUR_Q80_FOUR',
'TONSTAD_FOUR_Q80_FOURDTCHANGED',
'TONSTAD_FOUR_Q100_FOUR']

import datetime
for case in cases:
    start = datetime.datetime.now()
    lag_tre_multi(t_span=(0,179), filnamn_inn="../{}.hdf5".format(case), filnamn_ut="../{}.pickle".format(case))
    print("Ferdig med ", case, ", brukte " ,datetime.datetime.now()-start)
    