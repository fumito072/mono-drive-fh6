"""Read control evidence without trusting the recorder's completion label."""
import csv,json,struct
from pathlib import Path


def inspect_controls(folder):
    folder=Path(folder);raw=folder/'telemetry.raw';csvfile=folder/'telemetry.csv'
    result={'folder':str(folder),'raw_bytes':raw.stat().st_size if raw.exists() else None,
            'csv_bytes':csvfile.stat().st_size if csvfile.exists() else None,'raw_has_record':False,
            'csv_has_row':False,'csv_has_control_columns':False}
    if raw.exists():
        with raw.open('rb') as stream:
            signature=stream.read(8);header=stream.read(20)
            result['raw_signature_valid']=signature==b'FH6UDP1\n'
            if signature==b'FH6UDP1\n' and len(header)==20:
                _,_,size=struct.unpack('<QQI',header)
                result['first_packet_bytes']=size
                result['raw_has_record']=0<size<=65535 and len(stream.read(size))==size
    if csvfile.exists():
        with csvfile.open(encoding='utf-8-sig',newline='') as stream:
            table=csv.DictReader(stream);columns={str(x).lower() for x in table.fieldnames or []}
            result['csv_has_control_columns']=('steer' in columns and bool(columns&{'accel','throttle','brake'}))
            result['csv_has_row']=next(table,None) is not None
    try:metadata=json.loads((folder/'session.json').read_text(encoding='utf-8-sig'))
    except (OSError,ValueError):metadata={}
    result['status']=metadata.get('status');result['duration_s']=metadata.get('duration_seconds')
    if result['raw_has_record'] or (result['csv_has_row'] and result['csv_has_control_columns']):
        result['verdict']='control_records_present_requires_sync_validation'
    elif result['raw_bytes']==8 and result.get('raw_signature_valid') and not result['csv_has_row']:
        result['verdict']='header_only_no_recorded_controls'
    else:result['verdict']='missing_or_unknown_control_format'
    return result
