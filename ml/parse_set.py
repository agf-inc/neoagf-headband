print("=" * 60)#!/usr/bin/env python3
"""
EEG .set File Processor using MNE-Python

This script loads EEGLAB .set files, reports metadata, and exports filtered data to CSV.

Usage:
    python eeg_processor.py input_file.set --start_time 10.0 --end_time 60.0 --sensors Cz,Pz --output filtered_data.csv

Requirements:
    - mne
    - numpy
    - pandas
    - argparse
"""

import argparse
import sys
import numpy as np
import pandas as pd
import mne
from pathlib import Path


def load_and_inspect_set_file(filepath):
    """
    Load an EEGLAB .set file and report its contents.
    
    Parameters:
    -----------
    filepath : str
        Path to the .set file
        
    Returns:
    --------
    raw : mne.io.Raw
        Loaded raw EEG data
    """
    try:
        # Load the .set file
        raw = mne.io.read_raw_eeglab(filepath, preload=True, verbose=False)
        
        # If annotations exist, we can also convert them to events for analysis
        if raw.annotations is not None and len(raw.annotations) > 0:
            try:
                events_from_annot, event_dict = mne.events_from_annotations(raw, verbose=False)
                print(f"\nEvents converted from annotations: {len(events_from_annot)}")
                print(f"Event mapping: {event_dict}")
            except:
                print(f"\nCould not convert annotations to events")
        print(f"EEG FILE REPORT: {Path(filepath).name}")
        print("=" * 60)
        
        # Basic information
        print(f"Sampling frequency: {raw.info['sfreq']} Hz")
        print(f"Number of channels: {raw.info['nchan']}")
        print(f"Duration: {raw.times[-1]:.2f} seconds ({raw.times[-1]/60:.2f} minutes)")
        print(f"Number of samples: {len(raw.times)}")
        
        # Channel information
        print(f"\nChannel types:")
        ch_types = {}
        for ch_type in raw.get_channel_types():
            if ch_type in ch_types:
                ch_types[ch_type] += 1
            else:
                ch_types[ch_type] = 1
        for ch_type, count in ch_types.items():
            print(f"  {ch_type}: {count} channels")
        
        # List all channels
        print(f"\nAll channels ({len(raw.ch_names)}):")
        for i, ch_name in enumerate(raw.ch_names):
            print(f"  {i+1:2d}. {ch_name}")
        
        # Montage/electrode positions
        if raw.info['dig'] is not None:
            print(f"\nElectrode positions available: Yes")
            print(f"Number of digitized points: {len(raw.info['dig'])}")
        else:
            print(f"\nElectrode positions available: No")
        
        # Events and Annotations
        try:
            events = mne.find_events(raw, verbose=False)
            if len(events) > 0:
                print(f"\nEvents found: {len(events)}")
                unique_events = np.unique(events[:, 2])
                print(f"Unique event IDs: {unique_events}")
            else:
                print(f"\nEvents found: None")
        except:
            print(f"\nEvents found: None (no stim channels)")
        
        # Check for annotations
        if raw.annotations is not None and len(raw.annotations) > 0:
            print(f"\nAnnotations found: {len(raw.annotations)}")
            print(f"Annotation descriptions: {list(set(raw.annotations.description))}")
            
            # Show detailed annotation info
            print(f"\nAnnotation details:")
            for i, (onset, duration, description) in enumerate(zip(
                raw.annotations.onset, 
                raw.annotations.duration, 
                raw.annotations.description
            )):
                if i < 10:  # Show first 10 annotations
                    print(f"  {i+1:3d}. {onset:8.3f}s - {duration:6.3f}s : {description}")
                elif i == 10:
                    print(f"  ... ({len(raw.annotations) - 10} more annotations)")
                    break
                    
            # Summary by annotation type
            from collections import Counter
            ann_counts = Counter(raw.annotations.description)
            print(f"\nAnnotation summary:")
            for desc, count in ann_counts.items():
                print(f"  '{desc}': {count} occurrences")
        else:
            print(f"\nAnnotations found: None")
        
        # Data range
        data = raw.get_data()
        print(f"\nData statistics:")
        print(f"  Min value: {np.min(data):.2e}")
        print(f"  Max value: {np.max(data):.2e}")
        print(f"  Mean: {np.mean(data):.2e}")
        print(f"  Std: {np.std(data):.2e}")
        
        # Additional metadata from info
        print(f"\nAdditional metadata:")
        if 'subject_info' in raw.info and raw.info['subject_info']:
            print(f"  Subject info: {raw.info['subject_info']}")
        if 'experimenter' in raw.info and raw.info['experimenter']:
            print(f"  Experimenter: {raw.info['experimenter']}")
        if 'description' in raw.info and raw.info['description']:
            print(f"  Description: {raw.info['description']}")
        
        print("=" * 60)
        
        return raw
        
    except Exception as e:
        print(f"Error loading file: {e}")
        sys.exit(1)


def filter_and_export_data(raw, start_time=None, end_time=None, sensors=None, 
                          output_file="filtered_data.csv", target_sfreq=250):
    """
    Filter EEG data by time and sensors, resample to target frequency, and export to CSV.
    
    Parameters:
    -----------
    raw : mne.io.Raw
        Raw EEG data
    start_time : float, optional
        Start time in seconds
    end_time : float, optional
        End time in seconds
    sensors : list, optional
        List of sensor names to include
    output_file : str
        Output CSV filename
    target_sfreq : float
        Target sampling frequency (default: 250 Hz)
    """
    try:
        # Create a copy to avoid modifying original
        raw_filtered = raw.copy()
        
        # Filter by time
        if start_time is not None or end_time is not None:
            tmin = start_time if start_time is not None else 0
            tmax = end_time if end_time is not None else raw_filtered.times[-1]
            
            print(f"\nFiltering time range: {tmin:.2f}s to {tmax:.2f}s")
            raw_filtered.crop(tmin=tmin, tmax=tmax)
        
        # Filter by sensors
        if sensors is not None:
            # Convert to list if string
            if isinstance(sensors, str):
                sensors = [s.strip() for s in sensors.split(',')]
            
            # Check which sensors exist
            available_sensors = []
            missing_sensors = []
            
            for sensor in sensors:
                if sensor in raw_filtered.ch_names:
                    available_sensors.append(sensor)
                else:
                    missing_sensors.append(sensor)
            
            if missing_sensors:
                print(f"Warning: These sensors were not found: {missing_sensors}")
            
            if not available_sensors:
                print("Error: None of the requested sensors were found!")
                return
            
            print(f"Selecting sensors: {available_sensors}")
            raw_filtered.pick_channels(available_sensors)
        
        # Resample if necessary
        current_sfreq = raw_filtered.info['sfreq']
        if abs(current_sfreq - target_sfreq) > 0.1:  # Only resample if difference > 0.1 Hz
            print(f"Resampling from {current_sfreq} Hz to {target_sfreq} Hz")
            raw_filtered.resample(target_sfreq)
        else:
            print(f"No resampling needed (current: {current_sfreq} Hz)")
        
        # Get the filtered data
        data = raw_filtered.get_data().T  # Transpose to samples x channels
        times = raw_filtered.times
        
        # Create DataFrame
        df = pd.DataFrame(data, columns=raw_filtered.ch_names)
        df.insert(0, 'Time', times)  # Add time column as first column
        
        # Export to CSV
        df.to_csv(output_file, index=False)
        
        print(f"\nExported data:")
        print(f"  Output file: {output_file}")
        print(f"  Shape: {data.shape} (samples x channels)")
        print(f"  Time range: {times[0]:.3f}s to {times[-1]:.3f}s")
        print(f"  Sampling frequency: {raw_filtered.info['sfreq']:.1f} Hz")
        print(f"  Duration: {len(times) / raw_filtered.info['sfreq']:.2f} seconds")
        
        # Show first few rows
        print(f"\nFirst 5 rows of exported data:")
        print(df.head())
        
    except Exception as e:
        print(f"Error processing data: {e}")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="Process EEGLAB .set files with MNE-Python",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Just inspect the file
  python eeg_processor.py data.set
  
  # Extract 30 seconds of data from all channels at 250Hz
  python eeg_processor.py data.set --start_time 10 --end_time 40
  
  # Extract specific sensors
  python eeg_processor.py data.set --sensors "Cz,Pz,Oz" --output selected_channels.csv
  
  # Full filtering with custom output
  python eeg_processor.py data.set --start_time 5.0 --end_time 65.0 --sensors "Fp1,Fp2,Cz" --output custom_output.csv --target_freq 125
        """
    )
    
    parser.add_argument("input_file", help="Path to the input .set file")
    parser.add_argument("--start_time", type=float, help="Start time in seconds")
    parser.add_argument("--end_time", type=float, help="End time in seconds")
    parser.add_argument("--sensors", type=str, help="Comma-separated list of sensor names (e.g., 'Cz,Pz,Oz')")
    parser.add_argument("--output", default="filtered_data.csv", help="Output CSV filename (default: filtered_data.csv)")
    parser.add_argument("--target_freq", type=float, default=250, help="Target sampling frequency in Hz (default: 250)")
    parser.add_argument("--no_export", action="store_true", help="Only inspect file, don't export data")
    
    args = parser.parse_args()
    
    # Check if input file exists
    if not Path(args.input_file).exists():
        print(f"Error: File '{args.input_file}' does not exist!")
        sys.exit(1)
    
    # Load and inspect the file
    raw = load_and_inspect_set_file(args.input_file)
    
    # Export data if requested
    if not args.no_export:
        print("\n" + "="*60)
        print("PROCESSING DATA FOR EXPORT")
        print("="*60)
        
        filter_and_export_data(
            raw, 
            start_time=args.start_time,
            end_time=args.end_time,
            sensors=args.sensors,
            output_file=args.output,
            target_sfreq=args.target_freq
        )
    else:
        print("\nSkipping data export (--no_export flag used)")


if __name__ == "__main__":
    main()