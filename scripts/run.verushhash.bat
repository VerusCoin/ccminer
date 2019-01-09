@set PoolHost=na.luckpool.net
@set Port=3956
@set PublicVerusCoinAddress=RVjvbZuqSGLGDm1B9BFkbHWySPKEx4tfjQ
@set WorkerName=Worker1

@call :GET_CURRENT_DIR
@cd %THIS_DIR%
ccminer -a verus -o stratum+tcp://%PoolHost%:%Port% -u %PublicVerusCoinAddress%.%WorkerName%  %1 %2 %3 %4 %5 %6 %7 %8 %9
pause
@goto :EOF

:GET_CURRENT_DIR
@pushd %~dp0
@set THIS_DIR=%CD%
@popd
@goto :EOF



