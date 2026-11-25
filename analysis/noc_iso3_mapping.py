"""
IOC (NOC) 코드 ↔ ISO 3166-1 alpha-3 코드 매핑.

IOC(국제올림픽위원회)가 사용하는 국가 코드(NOC)는
ISO 3166-1 alpha-3과 상당수가 다르다.
DICL 데이터는 ISO3를 사용하므로 매핑이 필요하다.
"""

# NOC → ISO3 매핑 (IOC 코드가 ISO3와 다른 경우만)
# 동일한 코드(예: KOR, JPN, CHN)는 포함하지 않는다.
NOC_TO_ISO3 = {
    'ALG': 'DZA',  # Algeria
    'ANG': 'AGO',  # Angola
    'ARU': 'ABW',  # Aruba
    'BAH': 'BHS',  # Bahamas
    'BAN': 'BGD',  # Bangladesh
    'BAR': 'BRB',  # Barbados
    'BER': 'BMU',  # Bermuda
    'BHU': 'BTN',  # Bhutan
    'BIZ': 'BLZ',  # Belize
    'BOT': 'BWA',  # Botswana
    'BRN': 'BHR',  # Bahrain (NOC=BRN ≠ ISO3=BHR; ISO3 BRN is Brunei)
    'BRN': 'BHR',  # Bahrain (IOC NOC) — DICL ISO3 BRN is Brunei, must remap to BHR
    'BRU': 'BRN',  # Brunei
    'BUL': 'BGR',  # Bulgaria
    'BUR': 'BFA',  # Burkina Faso
    'CAM': 'KHM',  # Cambodia
    'CGO': 'COG',  # Congo
    'CHA': 'TCD',  # Chad
    'CHI': 'CHL',  # Chile
    'CRC': 'CRI',  # Costa Rica
    'CRO': 'HRV',  # Croatia
    'DEN': 'DNK',  # Denmark
    'ESA': 'SLV',  # El Salvador
    'FIJ': 'FJI',  # Fiji
    'GAM': 'GMB',  # Gambia
    'GER': 'DEU',  # Germany
    'GRE': 'GRC',  # Greece
    'GRN': 'GRD',  # Grenada
    'GUA': 'GTM',  # Guatemala
    'GUI': 'GIN',  # Guinea
    'HAI': 'HTI',  # Haiti
    'HON': 'HND',  # Honduras
    'INA': 'IDN',  # Indonesia
    'IRI': 'IRN',  # Iran
    'ISV': 'VIR',  # US Virgin Islands
    'KSA': 'SAU',  # Saudi Arabia
    'LAT': 'LVA',  # Latvia
    'LBA': 'LBY',  # Libya
    'LIB': 'LBN',  # Lebanon
    'MAD': 'MDG',  # Madagascar
    'MAS': 'MYS',  # Malaysia
    'MAW': 'MWI',  # Malawi
    'MGL': 'MNG',  # Mongolia
    'MRI': 'MUS',  # Mauritius
    'MTN': 'MRT',  # Mauritania
    'MYA': 'MMR',  # Myanmar
    'NCA': 'NIC',  # Nicaragua
    'NED': 'NLD',  # Netherlands
    'NEP': 'NPL',  # Nepal
    'NGR': 'NGA',  # Nigeria
    'NIG': 'NER',  # Niger
    'OMA': 'OMN',  # Oman
    'PAR': 'PRY',  # Paraguay
    'PHI': 'PHL',  # Philippines
    'PLE': 'PSE',  # Palestine
    'POR': 'PRT',  # Portugal
    'PUR': 'PRI',  # Puerto Rico
    'RSA': 'ZAF',  # South Africa
    'SAM': 'WSM',  # Samoa
    'SLO': 'SVN',  # Slovenia
    'SUD': 'SDN',  # Sudan
    'SUI': 'CHE',  # Switzerland
    'TAN': 'TZA',  # Tanzania
    'TGA': 'TON',  # Tonga
    'TOG': 'TGO',  # Togo
    'TPE': 'TWN',  # Chinese Taipei (Taiwan)
    'UAE': 'ARE',  # United Arab Emirates
    'URU': 'URY',  # Uruguay
    'VAN': 'VUT',  # Vanuatu
    'VIE': 'VNM',  # Vietnam
    'ZAM': 'ZMB',  # Zambia
    'ZIM': 'ZWE',  # Zimbabwe
}

# 역방향 매핑
ISO3_TO_NOC = {v: k for k, v in NOC_TO_ISO3.items()}


def noc_to_iso3(noc_code):
    """NOC 코드를 ISO3로 변환. 동일하면 그대로 반환."""
    return NOC_TO_ISO3.get(noc_code, noc_code)


def iso3_to_noc(iso3_code):
    """ISO3 코드를 NOC로 변환. 동일하면 그대로 반환."""
    return ISO3_TO_NOC.get(iso3_code, iso3_code)
