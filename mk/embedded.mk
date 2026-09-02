# Cross-compilation profile.
#
#   make PROFILE=embedded CROSS_COMPILE=arm-linux-gnueabihf-
#
# This targets embedded *Linux* (or any target with a libc and a BLAS), where
# it works today. Bare-metal is not reachable from here yet; see BLOCKERS.
#
# Only toggles belong in this file. It is included before the main Makefile
# resolves anything, so plain assignment here wins over the defaults there,
# while ?= still lets the command line win over this.

# Same caveat as the main Makefile: ?= cannot displace make's built-in CC/AR
# defaults, so test the origin. The command line still wins over this.
ifeq ($(origin CC),default)
  CC := $(CROSS_COMPILE)gcc
endif
ifeq ($(origin AR),default)
  AR := $(CROSS_COMPILE)ar
endif
RANLIB ?= $(CROSS_COMPILE)ranlib

# Most embedded toolchains have no OpenMP runtime, and the reservoir is
# usually small enough that thread setup costs more than it saves.
OPENMP := 0

# Size over speed: instruction cache is typically the binding constraint.
OPTIMIZE := -Os

# Keep the archive small: with -ffunction-sections the consumer's link can
# drop what it never calls, which matters because most of the training code
# is dead weight on a device that only runs inference.
EXTRA_CFLAGS += -ffunction-sections -fdata-sections
# Consumers should pair this with -Wl,--gc-sections when they link.

# BLAS := none
#   Uncomment once the reference kernels exist. Until then the cross build
#   still needs an OpenBLAS (or other CBLAS/LAPACKE) for the target, pointed
#   at with:
#     make PROFILE=embedded BLAS_CFLAGS=-I<sysroot>/include \
#          BLAS_LIBS="-L<sysroot>/lib -lopenblas -llapacke"

# ---------------------------------------------------------------------------
# BLOCKERS for a bare-metal build, in the order they will bite:
#
# 1. Variable-length arrays in the step path -- reservoir.c allocates
#    external_inputs[num_neurons], last_spikes[num_neurons] and
#    new_spikes[num_neurons] on the stack every step. At N=1000 that is ~24 KB
#    per call against a typical 4-16 KB bare-metal stack. These need to move
#    into the reservoir struct, allocated once at construction.
#
# 2. BLAS. The split is favourable: inference needs only dgemv, ddot and
#    dscal, all of which are a few lines of C each. dgemm, dger, dgesv and
#    dgeev are training and spectral-radius calibration only, so an
#    inference-only device build can drop them entirely. Note that
#    calc_spectral_radius_power_iteration() is still present and needs no
#    LAPACK, so calibration has a fallback if it must run on-device.
#
# 3. exit(EXIT_FAILURE) in neuron.c (x2) -- a library must not terminate its
#    host. These need to become error returns.
#
# 4. Heap use: ~100 malloc/calloc sites. Fine with an RTOS heap, but a
#    fully static build would need an arena or caller-provided storage.
#
# 5. stdio: ~66 fprintf/printf diagnostics across 7 files. Cheapest fix is a
#    SPIRES_LOG macro that compiles to nothing when stdio is unavailable.
# ---------------------------------------------------------------------------
