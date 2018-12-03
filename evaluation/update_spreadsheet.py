
import httplib2
import os
import sys
import ujson
import argparse

from apiclient import discovery
from oauth2client import client
from oauth2client import tools
from oauth2client.file import Storage
from pprint import pprint
from toolz.dicttoolz import get_in

SCOPES = 'https://www.googleapis.com/auth/spreadsheets'
CLIENT_SECRET_FILE = 'client_secret.json'
APPLICATION_NAME = 'ECG FiSci - Visual Search Tools'

#
# To get auth setup:
#
# - go to https://console.developers.google.com/apis/credentials
#
# - download the client ID named 'visual-search-tools'
# - rename to file downloaded to 'client_secret.json' and moved it to the root of the
#   evaluation module


def get_credentials():
    """Gets valid user credentials from storage.

    If nothing has been stored, or if the stored credentials are invalid,
    the OAuth2 flow is completed to obtain the new credentials.

    Returns:
        Credentials, the obtained credential.
    """
    home_dir = os.path.expanduser('~')
    credential_dir = os.path.join(home_dir, '.credentials')
    if not os.path.exists(credential_dir):
        os.makedirs(credential_dir)
    credential_path = os.path.join(
        credential_dir, 'sheets.googleapis.com-ecg-fisci-visual-search.json')

    store = Storage(credential_path)
    credentials = store.get()
    if not credentials or credentials.invalid:
        flow = client.flow_from_clientsecrets(CLIENT_SECRET_FILE, SCOPES)
        flow.user_agent = APPLICATION_NAME
        empty_flags = argparse.ArgumentParser(parents=[tools.argparser]).parse_args([])
        credentials = tools.run_flow(flow, store, empty_flags)
        print('Storing credentials to ' + credential_path)
    return credentials


def update_cells(spreadsheet_id, rangeName, values):
    """
    Examples:

        update_cells('log!R14', [[3.14159]])
        update_cells('log!R14:S14', [[1, 2]])
    """
    credentials = get_credentials()
    http = credentials.authorize(httplib2.Http())
    discoveryUrl = 'https://sheets.googleapis.com/$discovery/rest?version=v4'
    service = discovery.build('sheets', 'v4', http=http, discoveryServiceUrl=discoveryUrl)

    body = {
        'values': values
    }

    result = service.spreadsheets().values().update(
        spreadsheetId=spreadsheet_id,
        range=rangeName,
        valueInputOption='RAW',  # see https://developers.google.com/sheets/api/guides/values#writing
        body=body).execute()

    print('{0} cells updated.'.format(result.get('updatedCells')))


def copy_evaluation_to_spreadsheet(meta, evaluation, spreadsheet_id, sheet_name, row):
    for feature, feature_meta in meta.items():
        pprint(feature)
        data_keys = feature_meta['data_keys']
        pprint(data_keys)
        values = list(map(lambda key: get_in([key, 'accuracy', feature], evaluation), data_keys))
        pprint(values)
        col_begin = feature_meta['col_begin']
        col_end = feature_meta['col_end']
        rangeName = f'{sheet_name}!{col_begin}{row}:{col_end}{row}'
        pprint(rangeName)
        update_cells(spreadsheet_id, rangeName, [values])


def main():
    parser = argparse.ArgumentParser(
        description='Updates the accuracy section of the tracking spreadsheet.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('evaluation_file', nargs='?', type=argparse.FileType('r'), default=sys.stdin,
                        help='The evaluation file to process. Reads from stdin by default.')
    parser.add_argument('--spreadsheet_id', type=str,
                        default='1wHReorzGgHZ-8g1CeIhvCqLkJ9K4dUYz9heTABlIS7g',
                        help='The unique identifier of the evaluations spreadsheet.')
    parser.add_argument('--sheet_name', type=str, default='evaluations',
                        help='The name of the sheet to update.')
    parser.add_argument('--row', type=int, required=True,
                        help='The sheet row to update.')
    parser.add_argument('--evaluation_scheme', type=str, required=True,
                        choices=['as_is', 'by_frequency'],
                        help='The evaluation scheme used to produce the evaluation file.')

    args = parser.parse_args()

    meta = {
        'as_is': {
            'make': {'data_keys': ['1', '3', '5', '10'],
                     'col_begin': 'S', 'col_end': 'V'},
            'model': {'data_keys': ['1', '3', '5', '10'],
                      'col_begin': 'W', 'col_end': 'Z'},
            'color': {'data_keys': ['1', '3', '5', '10'],
                      'col_begin': 'AA', 'col_end': 'AD'},
            'body': {'data_keys': ['1', '3', '5', '10'],
                     'col_begin': 'AE', 'col_end': 'AH'},
            'year': {'data_keys': ['1', '3', '5', '10'],
                     'col_begin': 'AI', 'col_end': 'AL'}
        },
        'by_frequency': {
            'make': {'data_keys': ['1', '2', '3'],
                     'col_begin': 'AN', 'col_end': 'AP'},
            'model': {'data_keys': ['1', '2', '3'],
                      'col_begin': 'AQ', 'col_end': 'AS'},
            'color': {'data_keys': ['1', '2', '3'],
                      'col_begin': 'AT', 'col_end': 'AV'},
            'body': {'data_keys': ['1', '2', '3'],
                     'col_begin': 'AW', 'col_end': 'AY'},
            'year': {'data_keys': ['1', '2', '3'],
                     'col_begin': 'AZ', 'col_end': 'BB'}
        }
    }

    evaluation = ujson.load(args.evaluation_file)

    copy_evaluation_to_spreadsheet(
        meta[args.evaluation_scheme], evaluation, args.spreadsheet_id, args.sheet_name, args.row)


if __name__ == '__main__':
    main()
