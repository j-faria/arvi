import re

STARS = {
    '_51Peg': '51 Peg',
    'Barnard': 'GJ699',
    "Barnard's": 'GJ699',
    'Ross128': 'Ross  128',
    'Ross 128': 'Ross  128',
    # 
    'Teegarden': 'GAT1370',
    "Teegarden's Star": 'GAT1370',
    # 
    "Smethells 20": 'TIC464410508',
}


def translate(star, ngc=False, ic=False):
    # known translations
    if star in STARS:
        return STARS[star]

    # regex translations
    if ngc:
        NGC_match = re.match(r'NGC([\s\d]+)No([\s\d]+)', star)
        if NGC_match:
            cluster = NGC_match.group(1).replace(' ', '')
            target = NGC_match.group(2).replace(' ', '')
            return f'Cl* NGC {cluster} MMU {target}'
    if ic:
        IC_match = re.match(r'IC([\s\d]+)No([\s\d]+)', star)
        if IC_match:
            cluster = IC_match.group(1).replace(' ', '')
            target = IC_match.group(2).replace(' ', '')
            return f'Cl* IC {cluster} MMU {target}'

    return star
