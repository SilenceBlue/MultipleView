function KA=KernelAlignment(K1,K2)

KA=trace(K1'*K2)/sqrt(trace(K1*K1)*trace(K2'*K2));