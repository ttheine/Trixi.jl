"""
    init_mpi

Initialize MPI by calling `MPI.Initialized()`. The function will check if MPI is already initialized
and if yes, do nothing, thus it is safe to call it multiple times.
"""
function init_mpi()
  if MPI_INITIALIZED[]
    return nothing
  end

  if !MPI.Initialized()
    # MPI.THREAD_FUNNELED: Only main thread makes MPI calls
    provided = MPI.Init_thread(MPI.THREAD_FUNNELED)
    @assert provided >= MPI.THREAD_FUNNELED "MPI library with insufficient threading support"
  end

  # Initialize global MPI state
  MPI_RANK[] = MPI.Comm_rank(MPI.COMM_WORLD)
  MPI_SIZE[] = MPI.Comm_size(MPI.COMM_WORLD)
  MPI_IS_PARALLEL[] = MPI_SIZE[] > 1
  MPI_IS_SERIAL[] = !MPI_IS_PARALLEL[]
  MPI_IS_ROOT[] = MPI_IS_SERIAL[] || MPI_RANK[] == 0

  # Initialize methods for dispatching on parallel execution
  if MPI_IS_PARALLEL[]
    eval(:(mpi_parallel() = Val(true)))
  else
    eval(:(mpi_parallel() = Val(false)))
  end

  MPI_INITIALIZED[] = true

  return nothing
end


const MPI_INITIALIZED = Ref(false)
const MPI_RANK = Ref(-1)
const MPI_SIZE = Ref(-1)
const MPI_IS_PARALLEL = Ref(false)
const MPI_IS_SERIAL = Ref(true)
const MPI_IS_ROOT = Ref(true)


@inline mpi_comm() = MPI.COMM_WORLD

@inline mpi_rank(comm) = MPI.Comm_rank(comm)
@inline mpi_rank() = MPI_RANK[]

@inline n_mpi_ranks(comm) = MPI.Comm_size(comm)
@inline n_mpi_ranks() = MPI_SIZE[]

@inline is_parallel(comm) = n_mpi_ranks(comm) > 1
@inline is_parallel() = MPI_IS_PARALLEL[]

@inline is_serial(comm) = !is_parallel(comm)
@inline is_serial() = MPI_IS_SERIAL[]

@inline is_mpi_root(comm) = is_serial() || mpi_rank(comm) == 0
@inline is_mpi_root() = MPI_IS_ROOT[]

@inline mpi_root() = 0

@inline mpi_println(args...) = is_mpi_root() && println(args...)
@inline mpi_print(args...) = is_mpi_root() && print(args...)
