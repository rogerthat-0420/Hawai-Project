# Kernel build with Make using v++
# Requires Vitis (v++ in PATH) and PLATFORM env set to .xpfm name

VPP ?= v++
TARGET ?= hw_emu            # or hw_emu / sw_emu
PLATFORM ?= xilinx_u50_gen3x16_xdma_5_202210_1 # expect from env
KERNEL := forward
SRC := forward.cpp

BUILD_DIR := build
XO := $(BUILD_DIR)/$(KERNEL).xo
XCLBIN := $(BUILD_DIR)/$(KERNEL).xclbin

VPP_CFLAGS := -c -t $(TARGET) --platform $(PLATFORM) -k $(KERNEL)
VPP_LFLAGS := -l -t $(TARGET) --platform $(PLATFORM)

.PHONY: all xo xclbin clean

all: xclbin

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

xo: $(XO)
$(XO): $(SRC) | $(BUILD_DIR)
	@if [ -z "$(PLATFORM)" ]; then echo "Error: PLATFORM not set"; exit 1; fi
	$(VPP) $(VPP_CFLAGS) -o $@ $<

xclbin: $(XCLBIN)
$(XCLBIN): $(XO) | $(BUILD_DIR)
	$(VPP) $(VPP_LFLAGS) $< -o $@

clean:
	rm -rf $(BUILD_DIR)
