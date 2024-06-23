import re

STARS = {
    '_51Peg': '51 Peg',
    'Barnard': 'GJ699',
    "Barnard's": 'GJ699',
    'Ross128': 'Ross  128',
}


def translate(star):
    # known translations
    if star in STARS:
        return STARS[star]

    # regex translations
    NGC_match = re.match(r'NGC([\s\d]+)No([\s\d]+)', star)
    if NGC_match:
        cluster = NGC_match.group(1).replace(' ', '')
        target = NGC_match.group(2).replace(' ', '')
        return f'Cl* NGC {cluster} MMU {target}'

    return star
