# Calcul des moments théoriques



def autocovariance(a0,a1,b1,n=2):
    gamma = 2*a1*(1-a1*b1-b1**2)*(a1+b1)**(n-1)/(1-3*a1**2-2*a1*b1-b1**2)
    return gamma


def a(j):
    if j <= 1:
        return 1
    else:
        return (2*j-1)*a(j-1)

def factorial(n):
    if n <= 1:
        return 1
    else:
        return n*factorial(n-1)
    
def binom(n,k):
    return factorial(n)/(factorial(k)*factorial(n-k))

def mu(a1,b1,n):
    s = 0
    for j in range(0,n+1):
        s += a(j) * binom(n,j) * a1**j * b1**(n-j)
    return(s)

def moments(a0,a1,b1,m = 3):
    momentsTab = [1]
    for m1 in range(1,m+1):
        s = 0
        for n in range(m1):
            s += (momentsTab[n] * a0**(m1-n) * binom(m1,n) * mu(a1,b1,n)/a(n))
        s *= a(m1)
        s /= (1-mu(a1,b1,m1))
        momentsTab.append(s)

    for n in range(2,m+1):
        momentsTab[n] /= momentsTab[1]**n
    return momentsTab[1:]
    

def moments1(a0,a1,b1):
    M2 = a0/(1-a1-b1)
    M4 = 3 + 6*a1**2/(1-3*a1**2-2*a1*b1-b1**2)
    M6 = 15*(1-a1-b1)**3*(1+3*(a1+b1)/(1-a1-b1)+3*(1+2*(a1+b1)/(1-a1-b1))*(b1**2+2*a1*b1+3*a1**2)/(1-3*a1**2-2*a1*b1-b1**2))/(1-15*a1**3-9*a1**2*b1-3*a1*b1**2-b1**3)
    return M2, M4, M6
