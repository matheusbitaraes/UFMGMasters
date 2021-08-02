rm(list=ls())
library("kernlab")
library(mlbench)
library('plot3D')

s1<-0.5
s2<-0.5
N<-1000 #numeros pares

p <- mlbench.spirals(N,1,0.05)
ic1<-which(p[[2]]==1)
ic2<-which(p[[2]]==2)
xall<-as.matrix(p[[1]])
xc1<-xall[ic1,]
xc2<-xall[ic2,]

plot(xc1[,1],xc1[,2],type = 'p',col = 'blue', xlim=c(min(xall),max(xall)),ylim=c(min(xall),max(xall)))
par(new=T)
plot(xc2[,1],xc2[,2],type = 'p',col = 'red', xlim=c(min(xall),max(xall)),ylim=c(min(xall),max(xall)))

xin=rbind(xc1,xc2)
yin=rbind(matrix(-1,N/2,1),matrix(1,N/2,1))

plot(xc1[,1],xc1[,2],type = 'p',col = 'red', xlim=c(min(xall),max(xall)),ylim=c(min(xall),max(xall)))
plot(xc2[,1],xc2[,2],col = 'blue')

svmtrein <- ksvm(xin,yin,type='C-bsvc',kernel='rbfdot',kpar=list(sigma=0.5),C=10)

yhat<-predict(svmtrein,xin,type='response')

a<-alpha(svmtrein)
ai<-SVindex(svmtrein)
nsvec=nSV(svmtrein)
points(xin[ai,1],xin[ai,2],col="green")

# hiperplano em cada ponrto de um grid
seqi<-seq(-1.5,1.5,0.05)
seqj<-seq(-1.5,1.5,0.05)
M1<-matrix(0,nrow=length(seqi),ncol=length(seqj))

ci<-0
for (i in seqi){
  ci<-ci+1
  cj<-0
  for(j in seqj){
    cj<-cj+1
    M1[ci,cj]<-predict(svmtrein,as.matrix(cbind(i,j)),type="response")
  }
}

# plotando a separação
plot(xc1[,1],xc1[,2],type = 'p',col = 'blue', xlim=c(min(xall),max(xall)),ylim=c(min(xall),max(xall)))
par(new=T)
plot(xc2[,1],xc2[,2],type = 'p',col = 'red', xlim=c(min(xall),max(xall)),ylim=c(min(xall),max(xall)))
par(new=T)
contour(seqi,seqj,M1,nlevels = 0,xlim=c(min(xall),max(xall)),ylim=c(min(xall),max(xall)))

persp3D(seqi,seqj,M1,counter=T,theta=55,phi=30,r=40,d=0.1,expand=0.5,ltheta=90,lphi=180,shade=0.4,ticktype="detailed",nticks=5)

#matrix de kernel
MK <- kernelMatrix(rbfdot(sigma=0.5),xin)
image(MK)
