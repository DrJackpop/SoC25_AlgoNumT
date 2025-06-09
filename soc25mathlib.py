def pair_gcd(a :int ,b :int):
    while b!=0 : 
        a,b=b,a%b
    return int(a)

def pair_egcd(a: int, b: int):
    if a%b==0 :
        return 0,1,b
    else : 
        x,y,d=pair_egcd(b,a%b)
        return y,x-a//b * y,d
        

        

def gcd(*args :int):
    g=args[0]   
    for b in args:
        g=pair_gcd(g,b)
    return g

def pair_lcm(a: int, b: int) :
    return int(a*b//gcd(a,b))

def lcm(*args : int):
    l=args[0]
    for b in args:
        l=pair_lcm(l,b)
    return l

def are_relatively_prime(a : int,b :int):
    if pair_gcd(a,b)==1 : 
        return True 
    return False

def mod_inv(a: int, n: int):
    if pair_gcd(a,n)!=1 : raise Exception("a and n not coprime, no inverse modulo exists")
    x,y,d=pair_egcd(n,a)
    if x>0 : return n+y
    return y

def crt(a: list[int],n: list[int]):
    N,x=1,0
    for i in n : N*=i
    for i,j in enumerate(n) :
        x+=mod_inv(N//j,j)*N//j*a[i]
    return int(x%N)

def is_quadratic_residue_prime(a: int, p: int):
    x=pow(a,int((p-1)/2),p)
    if x>1 : return -1
    else : return x
    
def is_quadratic_residue_prime_power(a: int, p: int, e: int):
    if a%p!=0 :
        return is_quadratic_residue_prime(a,p)
    else :  
        n = 0
        while a%p == 0 : 
            a//=p
            n+=1
        return int(n%2==0 & max(is_quadratic_residue_prime(a,p),0))
    