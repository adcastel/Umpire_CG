#make clean && make
make

#Node selection
OBJ=CG_Umpire 
#MATRIX
export ROW_PTR=0
export COL_IDX=0
export VALUES=0

export B=0
export X=0

#VECTORS
export R=0
export Z=0
export P=0
export AP=0

#CHECK
export AX=0

./$OBJ matrix.csr
./$OBJ audikw_1.mtx 
./$OBJ /root/matrix/audikw_1/audikw_1.mtx 
#./$OBJ matrix.csr
