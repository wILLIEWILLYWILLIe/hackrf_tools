#!/bin/bash

# transmit.sh
# Adaptive Bash script to generate and transmit a tone, chirp, hopping, or square block signal using HackRF

# Exit immediately if a command exits with a non-zero status
set -e

# Function to display usage information
function usage() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  -m, --mode           Signal mode: tone, chirp, hopping, or square_block (required)"
    echo ""
    echo "  # For Tone Mode:"
    echo "  -f, --frequency      Tone frequency in Hz (e.g., 100000 for 100 kHz)"
    echo ""
    echo "  # For Chirp Mode:"
    echo "      --start-freq     Chirp start frequency in Hz"
    echo "      --end-freq       Chirp end frequency in Hz"
    echo ""
    echo "  # For Hopping Mode:"
    echo "      --start-freq     Hopping start frequency in Hz"
    echo "      --end-freq       Hopping end frequency in Hz"
    echo "      --hop-step       Frequency step between hops in Hz"
    echo "      --hop-duration   Duration of each hop in seconds"
    echo ""
    echo "  # For Square Block Mode:"
    echo "      --block-freq     Block center frequency in Hz"
    echo "      --bandwidth      Bandwidth of the block in Hz"
    echo "      --block-duration Duration of each block in seconds"
    echo "      --time-gap       Duration of silence between blocks in seconds"
    echo ""
    echo "  # Common Options:"
    echo "  -s, --sample-rate    Sample rate in Hz (default: 2000000)"
    echo "  -d, --duration       Total duration in seconds (default: 5)"
    echo "  -a, --amplitude      Amplitude (0 < amplitude <= 1, default: 0.5)"
    echo "  -c, --center-freq    Center frequency in Hz (default: 2450000000)"
    echo "  -x, --tx-gain        TX VGA gain in dB (default: 30)"
    echo "  -l, --lna-gain       LNA gain in dB (default: 40)"
    echo "  -A, --antenna        Antenna port: 1=on, 0=off (default: 1)"
    echo "  -h, --help           Display this help message"
    echo ""
    echo "Examples:"
    echo "  # Transmit a tone"
    echo "  $0 -m tone -f 100000 -s 2000000 -d 5 -a 0.5 -c 2450000000 -x 30 -l 40 -A 1"
    echo ""
    echo "  # Transmit a chirp"
    echo "  $0 -m chirp --start-freq 90000 --end-freq 110000 -s 2000000 -d 5 -a 0.5 -c 2450000000 -x 30 -l 40 -A 1"
    echo ""
    echo "  # Transmit a hopping signal"
    echo "  $0 -m hopping --start-freq 50000 --end-freq 150000 --hop-step 20000 --hop-duration 1 -s 2000000 -d 10 -a 0.7 -c 2450000000 -x 35 -l 45 -A 1"
    echo ""
    echo "  # Transmit a square block signal"
    echo "  $0 -m square_block --block-freq 2400000000 --bandwidth 50000 --block-duration 2 --time-gap 1 -s 2000000 -d 20 -a 0.6 -c 2450000000 -x 40 -l 50 -A 1"
    exit 1
}

# Function to display info messages
function echo_info {
    echo -e "\e[34m[INFO]\e[0m $1"
}

# Function to display error messages
function echo_error {
    echo -e "\e[31m[ERROR]\e[0m $1" >&2
}

# Initialize variables with default values

MODE="square_block"
FREQUENCY=1000000
START_FREQ=80000
END_FREQ=120000
HOP_STEP=80000
HOP_DURATION=0.2

BLOCK_FREQ=10000000
BANDWIDTH=10000000
BLOCK_DURATION=5e-1
TIME_GAP=1e-5
SAMPLE_RATE=20000000
DURATION=5e-1
AMPLITUDE=1
CENTER_FREQ=2450000000
TX_GAIN=400
LNA_GAIN=400
ANTENNA=1

# Parse command-line arguments using getopt for both short and long options
ARGS=$(getopt -o m:f:s:d:a:c:x:l:A:h --long mode:,frequency:,start-freq:,end-freq:,hop-step:,hop-duration:,block-freq:,bandwidth:,block-duration:,time-gap:,sample-rate:,duration:,amplitude:,center-freq:,tx-gain:,lna-gain:,antenna:,help -n 'transmit.sh' -- "$@")
if [ $? != 0 ] ; then usage ; fi

eval set -- "$ARGS"

while true; do
    case "$1" in
        -m|--mode)
            MODE="$2"
            shift 2
            ;;
        -f|--frequency)
            FREQUENCY="$2"
            shift 2
            ;;
        --start-freq)
            START_FREQ="$2"
            shift 2
            ;;
        --end-freq)
            END_FREQ="$2"
            shift 2
            ;;
        --hop-step)
            HOP_STEP="$2"
            shift 2
            ;;
        --hop-duration)
            HOP_DURATION="$2"
            shift 2
            ;;
        --block-freq)
            BLOCK_FREQ="$2"
            shift 2
            ;;
        --bandwidth)
            BANDWIDTH="$2"
            shift 2
            ;;
        --block-duration)
            BLOCK_DURATION="$2"
            shift 2
            ;;
        --time-gap)
            TIME_GAP="$2"
            shift 2
            ;;
        -s|--sample-rate)
            SAMPLE_RATE="$2"
            shift 2
            ;;
        -d|--duration)
            DURATION="$2"
            shift 2
            ;;
        -a|--amplitude)
            AMPLITUDE="$2"
            shift 2
            ;;
        -c|--center-freq)
            CENTER_FREQ="$2"
            shift 2
            ;;
        -x|--tx-gain)
            TX_GAIN="$2"
            shift 2
            ;;
        -l|--lna-gain)
            LNA_GAIN="$2"
            shift 2
            ;;
        -A|--antenna)
            ANTENNA="$2"
            shift 2
            ;;
        -h|--help)
            usage
            ;;
        --)
            shift
            break
            ;;
        *)
            break
            ;;
    esac
done

# Validate mode
if [[ -z "$MODE" ]]; then
    echo_error "Signal mode is required."
    usage
fi

if [[ "$MODE" != "tone" && "$MODE" != "chirp" && "$MODE" != "hopping" && "$MODE" != "square_block" ]]; then
    echo_error "Invalid mode. Choose 'tone', 'chirp', 'hopping', or 'square_block'."
    usage
fi

# Validate amplitude
if (( $(echo "$AMPLITUDE <= 0" | bc -l) )) || (( $(echo "$AMPLITUDE > 1" | bc -l) )); then
    echo_error "Amplitude must be greater than 0 and less than or equal to 1."
    exit 1
fi

# Validate antenna
if [[ "$ANTENNA" != "0" && "$ANTENNA" != "1" ]]; then
    echo_error "Antenna must be either 1 (on) or 0 (off)."
    exit 1
fi

# Validate sample rate
if (( SAMPLE_RATE < 100000 )) || (( SAMPLE_RATE > 20000000 )); then
    echo_error "Sample rate must be between 100,000 Hz and 20,000,000 Hz (2 MS/s)."
    exit 1
fi

# Validate frequency parameters based on mode
if [[ "$MODE" == "tone" ]]; then
    if (( FREQUENCY <= 0 )); then
        echo_error "Tone frequency must be a positive number."
        exit 1
    fi
elif [[ "$MODE" == "chirp" ]]; then
    if (( START_FREQ <= 0 )) || (( END_FREQ <= 0 )); then
        echo_error "Chirp start and end frequencies must be positive numbers."
        exit 1
    fi
    if (( START_FREQ >= END_FREQ )); then
        echo_error "Chirp start frequency must be less than end frequency."
        exit 1
    fi
elif [[ "$MODE" == "hopping" ]]; then
    if (( START_FREQ <= 0 )) || (( END_FREQ <= 0 )) || (( HOP_STEP <= 0 )) || (( HOP_DURATION <= 0 )); then
        echo_error "Hopping start/end frequencies, hop step, and hop duration must be positive numbers."
        exit 1
    fi
    if (( START_FREQ >= END_FREQ )); then
        echo_error "Hopping start frequency must be less than end frequency."
        exit 1
    fi
elif [[ "$MODE" == "square_block" ]]; then
    if (( BLOCK_FREQ < 0 )) || (( BANDWIDTH <= 0 )) || (( BLOCK_DURATION <= 0 )) || (( TIME_GAP < 0 )); then
        echo_error "Square block center frequency, bandwidth, block duration must be positive numbers and time gap cannot be negative."
        exit 1
    fi
    if (( BANDWIDTH > SAMPLE_RATE / 2 )); then
        echo_error "Bandwidth cannot exceed Nyquist frequency (sample_rate / 2)."
        exit 1
    fi
fi

# Define output filenames based on mode
if [[ "$MODE" == "tone" ]]; then
    OUTPUT_FILE="IQs_bins/tone_iq.bin"
    PYTHON_SCRIPT="./packages/generate_tone.py"
elif [[ "$MODE" == "chirp" ]]; then
    OUTPUT_FILE="IQs_bins/chirp_iq.bin"
    PYTHON_SCRIPT="./packages/generate_chirp.py"
elif [[ "$MODE" == "hopping" ]]; then
    OUTPUT_FILE="IQs_bins/hopping_iq.bin"
    PYTHON_SCRIPT="./packages/generate_hopping.py"
elif [[ "$MODE" == "square_block" ]]; then
    OUTPUT_FILE="IQs_bins/square_block_iq.bin"
    PYTHON_SCRIPT="./packages/generate_square_block.py"
fi

# Check if the required Python script exists
if [[ ! -f "$PYTHON_SCRIPT" ]]; then
    echo_error "Python script $PYTHON_SCRIPT not found in the current directory."
    exit 1
fi

# Execute the Python script based on mode
echo_info "Generating IQ data using $PYTHON_SCRIPT..."

if [[ "$MODE" == "tone" ]]; then
    python3 $PYTHON_SCRIPT -f $FREQUENCY -s $SAMPLE_RATE -d $DURATION -a $AMPLITUDE -o $OUTPUT_FILE
elif [[ "$MODE" == "chirp" ]]; then
    python3 $PYTHON_SCRIPT --start-freq $START_FREQ --end-freq $END_FREQ -s $SAMPLE_RATE -d $DURATION -a $AMPLITUDE -o $OUTPUT_FILE
elif [[ "$MODE" == "hopping" ]]; then
    python3 $PYTHON_SCRIPT --start-freq $START_FREQ --end-freq $END_FREQ --hop-step $HOP_STEP --hop-duration $HOP_DURATION -s $SAMPLE_RATE -d $DURATION -a $AMPLITUDE -o $OUTPUT_FILE
elif [[ "$MODE" == "square_block" ]]; then
    python3 $PYTHON_SCRIPT --center-freq $BLOCK_FREQ --bandwidth $BANDWIDTH --block-duration $BLOCK_DURATION --time-gap $TIME_GAP -s $SAMPLE_RATE -d $DURATION -a $AMPLITUDE -o $OUTPUT_FILE
fi

# Verify that the IQ file was created
if [ ! -f "$OUTPUT_FILE" ]; then
    echo_error "IQ data file $OUTPUT_FILE was not created."
    exit 1
fi

echo_info "IQ data generated and saved to $OUTPUT_FILE."

# Transmit the IQ data using hackrf_transfer
echo_info "Transmitting IQ data using hackrf_transfer..."
FREQUENCIES=(
    1500000000
    2000000000
)
FREQUENCIES=(
    5770000000
    5790000000
    # 5800000000
    # 5810000000
)
while true; do
    for FREQ in "${FREQUENCIES[@]}"; do
        hackrf_transfer \
            -t "$OUTPUT_FILE" \
            -f "$FREQ" \
            -s "$SAMPLE_RATE" \
            -x "$TX_GAIN" \
            -a "$ANTENNA" \
            -l "$LNA_GAIN"
    done
done
echo_info "Transmission complete."

# Optional: Clean up by removing the IQ file
read -p "Do you want to remove the IQ data file ($OUTPUT_FILE)? (y/n): " CLEANUP
if [[ "$CLEANUP" == "y" || "$CLEANUP" == "Y" ]]; then
    echo_info "Removing $OUTPUT_FILE..."
    rm $OUTPUT_FILE
    echo_info "Cleanup complete."
fi

exit 0