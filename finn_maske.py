from pathlib import Path
from bs4 import BeautifulSoup

mapper = [{"Exp":"TONSTAD_FOUR",	"macro":"FOUR.P002.mkr",	"folder":"Q100_FOUR"},
{"Exp":"TONSTAD_FOUR",	"macro":"FOUR.P002.mkr",	"folder":"Q100_FOUR DT"},
{"Exp":"TONSTAD_FOUR",	"macro":"FOUR.P002.mkr",	"folder":"Q100 DTCHANGED"},
{"Exp":"TONSTAD_FOUR",	"macro":"FOUR.P002.mkr",	"folder":"Q20_FOUR TRIALONE"},
{"Exp":"TONSTAD_FOUR",	"macro":"FOUR.P002.mkr",	"folder":"Q20_FOUR CHECK"},
{"Exp":"TONSTAD_FOUR",	"macro":"FOUR.P002.mkr",	"folder":"Q20_FOUR REPEAT"},
{"Exp":"TONSTAD_FOUR",	"macro":"FOUR.P002.mkr",	"folder":"Q20_FOURDTDECREASE"},
{"Exp":"TONSTAD_FOUR",	"macro":"FOUR.P002.mkr",	"folder":"Q40_FOUR"},
{"Exp":"TONSTAD_FOUR",	"macro":"FOUR.P002.mkr",	"folder":"Q40_REPEAT"},
{"Exp":"TONSTAD_FOUR",	"macro":"FOUR.P002.mkr",	"folder":"Q60_FOUR"},
{"Exp":"TONSTAD_FOUR",	"macro":"FOUR.P002.mkr",	"folder":"Q60_FOUR REPEAT"},
{"Exp":"TONSTAD_FOUR",	"macro":"FOUR.P002.mkr",	"folder":"Q80_FOUR"},
{"Exp":"TONSTAD_FOUR",	"macro":"FOUR.P002.mkr",	"folder":"Q80_FOUR TRIAL"},
{"Exp":"TONSTAD_FOUR",	"macro":"FOUR.P002.mkr",	"folder":"Q80_FOURDTCHANGED"},
{"Exp":"Tonstad_THREE",	"macro":"mask1.P002.mkr",	"folder":"Q20_THREE"},
{"Exp":"Tonstad_THREE",	"macro":"mask2.P003.mkr",	"folder":"Q40_THREE"},
{"Exp":"Tonstad_THREE",	"macro":"mask2.P003.mkr",	"folder":"Q40_THREE_EXTRA"},
{"Exp":"Tonstad_THREE",	"macro":"mask4.P005.mkr",	"folder":"Q40_THREE_FINAL"},
{"Exp":"Tonstad_THREE",	"macro":"mask3.P004.mkr",	"folder":"Q40_THREE_REPEAT"},
{"Exp":"Tonstad_THREE",	"macro":"mask2.P003.mkr",	"folder":"Q60_THREE"},
{"Exp":"Tonstad_THREE",	"macro":"mask2.P003.mkr",	"folder":"Q80_THREE"},
{"Exp":"Tonstad_THREE",	"macro":"mask3.P004.mkr",	"folder":"Q80_THREE_EXTRA"},
{"Exp":"Tonstad_THREE",	"macro":"mask5.P006.mkr",	"folder":"Q80_THREE_EXTRA2"},
{"Exp":"Tonstad_THREE",	"macro":"mask4.P005.mkr",	"folder":"Q100_THREE"},
{"Exp":"Tonstad_THREE",	"macro":"mask4.P005.mkr",	"folder":"Q100_THREE_EXTRA"},
{"Exp":"Tonstad_THREE",	"macro":"mask5.P006.mkr",	"folder":"Q100_THREE_EXTRA2"},
{"Exp":"TONSTAD_TWO",	"macro":"prosess_TWO.P002.mkr",	"folder":"Q20_TWO"},
{"Exp":"TONSTAD_TWO",	"macro":"prosess_TWO.P002.mkr",	"folder":"Q40_TWO"},
{"Exp":"TONSTAD_TWO",	"macro":"prosess_TWO.P002.mkr",	"folder":"Q60_TWO"},
{"Exp":"TONSTAD_TWO",	"macro":"prosess_TWO.P002.mkr",	"folder":"Q80_TWO"},
{"Exp":"TONSTAD_TWO",	"macro":"prosess_TWO.P002.mkr",	"folder":"Q100_TWO"},
{"Exp":"TONSTAD_TWO",	"macro":"prosess_TWO.P002.mkr",	"folder":"Q120_TWO"},
{"Exp":"TONSTAD_TWO",	"macro":"prosess_TWO.P002.mkr",	"folder":"Q140_TWO"}]


ny={}
for m in mapper:
    e = Path("/mnt/g/Experiments11/").joinpath(m['Exp']).joinpath("Settings").joinpath(m['macro'])
    with open(e) as f:
        data = f.read()
    bs_data = BeautifulSoup(data,'xml')

    ribbefil = bs_data.MaskData.contents[3].name
    e.joinpath(ribbefil)
    ny[m['folder']] = dict(exp = m['Exp'], rib=ribbefil)