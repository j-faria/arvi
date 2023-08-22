STARS = {
    'Barnard': 'GJ699',
    "Barnard's": 'GJ699',
}


def translate(star):
    if star in STARS:
        return STARS[star]
    return star
