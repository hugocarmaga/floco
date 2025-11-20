__pkg_name__ = 'floco'
__title__ = 'Floco'
__version__ = '1.0.0'
__author__ = 'Hugo Magalhães, Jonas Weber, Gunnar W. Klau, Tobias Marschall, Timofey Prodanov'
__license__ = 'MIT'

def long_version():
    authors = __author__.split(', ')
    if len(authors) > 1:
        authors_str = ', '.join(authors[:-1]) + ' & ' + authors[-1]
    else:
        authors_str = authors[0]
    return '{} v{}\nCreated by {}'.format(__title__, __version__, authors_str)