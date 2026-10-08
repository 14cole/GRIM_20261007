"""BoR sample assembly with optional shared, float64 application storage."""
from ghost_backend.twod.samples import (
    BOR_FIELDS, LabeledSamples, SampleChunks, SampleTable, sample_buffer,
    sample_column, sorted_samples,
)
import math
import numpy as np


def channel_buffers(frequency_count, aspects, expand_to_360):
    count = len(aspects)
    if expand_to_360:
        count += sum(0.0 < angle < 180.0 for angle in aspects)
    return {channel: sample_buffer(frequency_count * count, fields=BOR_FIELDS)
            for channel in ('VV', 'HH')}


def finish_channels(channels, expand_to_360):
    """Keep the legacy frequency/channel/aspect order, including duplicate inputs.

    Channel rows stay unlabeled, as in the public list API; the combined view
    adds polarization labels without retaining another copy of the numbers.
    """
    compact = all(isinstance(rows, SampleTable) for rows in channels.values())
    if expand_to_360:
        for channel, rows in channels.items():
            for index in range(len(rows)):
                source = rows[index]
                angle = float(source['theta_inc_deg'])
                if 0.0 < angle < 180.0:
                    mirrored = dict(source)
                    mirrored['theta_inc_deg'] = mirrored['theta_scat_deg'] = 360.0 - angle
                    rows.append(mirrored)
            channels[channel] = sorted_samples(rows, keys=('frequency_ghz', 'theta_inc_deg'))
    if compact:
        combined = SampleChunks()
        for channel in ('VV', 'HH'):
            combined.extend(LabeledSamples(channels[channel], polarization=channel))
        return sorted_samples(combined, keys=('frequency_ghz', 'polarization', 'theta_inc_deg'))
    combined = [dict(row, polarization=channel)
                for channel in ('VV', 'HH') for row in channels[channel]]
    combined.sort(key=lambda row: (row['frequency_ghz'], row['polarization'], row['theta_inc_deg']))
    return combined


def nonfinite_sample_count(channels):
    count = 0
    keys = ('rcs_linear', 'rcs_amp_real', 'rcs_amp_imag')
    for rows in channels.values():
        first = sample_column(rows, keys[0])
        if first is None:
            count += sum(not all(math.isfinite(float(row[key])) for key in keys) for row in rows)
            continue
        finite = np.isfinite(first)
        for key in keys[1:]:
            finite &= np.isfinite(sample_column(rows, key))
        count += int(np.count_nonzero(~finite))
    return count
