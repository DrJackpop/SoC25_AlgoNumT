# Assignment 1

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
    
    
    # Assignment 2
import random

def floor_sqrt(x:int):
    k = (x.bit_length()-1)//2
    m = 1<<k
    for i in range(k-1,-1,-1):
        if (m+(1<<i))**2 <=x : m+=1<<i
    return m


def floor_nth_root(a: int, b: int):
    k = (a.bit_length()-1)//b
    m = 1<<k
    for i in range(k-1,-1,-1):
        if (m +(1<<i))**b <= a:
            m += 1 << i
    return m
def is_perfect_power(x: int) :
    l = x.bit_length()
    for b in range(2, l+1):
        a = floor_nth_root(x, b)
        if a**b == x or (a+1)**b == x: 
            return True
    return False

def is_prime(n: int, k=5) -> bool:
    if n==2 or n==3: return True
    d, r = n - 1, 0
    while d % 2 == 0:
        d //= 2
        r += 1
    for _ in {2,3,5,7,11,13}:
        a = random.randrange(2, n - 1)
        x = pow(a, d, n)
        if x == 1 or x == n - 1:
            continue
        for _ in range(r - 1):
            x = pow(x, 2, n)
            if x == n - 1: break
        else:
            return False
    return True

def gen_prime(m:int):
    p=m
    while not is_prime(p):
        p=random.randint(2,m)
    return p

def gen_k_bit_prime(k: int):
    p=2**k-1
    while not is_prime(p):
        p=random.randint(2**{k-1},2**k)
    return p

def factor(n:int):
    factor=[]
    if n==1: return factor
    if is_prime(n): return [(n,1)]
    d=2
    while d*d<=n:
        if n % d==0 and is_prime(d): 
            count=0
            while n % d==0:
                count+=1
                n//=d
            factor.append((d,count))
        if(d==2) : d-=1
        d+=2
    if n>1 : factor.append((n,1))
    return factor

def euler_phi(n:int):
    if n==1:return n
    fac=factor(n)
    euler=1
    for i,j in fac:
        euler*=(i**(j-1))*(i-1)
    return euler

class QuotientPolynomialRing:
    def __init__(self,poly:list[int],pi_gen:list[int]):
        self.pi_generator=pi_gen
        self.element=self.modulo(poly)
    
    def modulo(self,poly):
        while len(poly)>len(self.pi_generator)-1:
            fac=poly[-1]
            for i in range(1,len(self.pi_generator)+1):
                poly[-i]=poly[-i]-fac*self.pi_generator[-i]
            poly.pop()
        return poly+[0]*max(0,(len(self.pi_generator)-len(poly)-1))
    
    @staticmethod
    def Add(poly1: 'QuotientPolynomialRing', poly2: 'QuotientPolynomialRing'):
        if poly1.pi_generator != poly2.pi_generator:
            raise ValueError('Different pi_generators')
        add = []
        n = len(poly1.pi_generator)
        poly1.element+=[0]*max(0,(n-len(poly1.element)))
        poly2.element+=[0]*max(0,(n-len(poly2.element)))
        for _ in range(n):
            add.append(poly1.element[_]+poly2.element[_])
        return QuotientPolynomialRing(add, poly1.pi_generator)

    @staticmethod
    def Sub(poly1: 'QuotientPolynomialRing', poly2: 'QuotientPolynomialRing'):
        if poly1.pi_generator != poly2.pi_generator:
            raise ValueError('Different pi_generators')
        sub = []
        for _ in range(len(poly1.pi_generator)):
            sub.append(poly1.element[_]-poly2.element[_])
        return QuotientPolynomialRing(sub, poly1.pi_generator)
    
    @staticmethod
    def Mul(poly1: 'QuotientPolynomialRing', poly2: 'QuotientPolynomialRing'):
        if poly1.pi_generator != poly2.pi_generator:
            raise ValueError('Different pi_generators')    
        prod = [0] * (len(poly2.element)+len(poly1.element)-1) 
        for i in range(len(poly1.element)):
           for j in range(len(poly2.element)):
                prod[i + j] += poly1.element[i] * poly2.element[j]
        return QuotientPolynomialRing(prod, poly1.pi_generator)
    
    @staticmethod
    def GCD(poly1: 'QuotientPolynomialRing', poly2: 'QuotientPolynomialRing'):
        if poly1.pi_generator != poly2.pi_generator:
            raise ValueError('Different pi_generators')

        a = poly1.element.copy()
        b = poly2.element.copy()
        
        while a[-1]==0:a.pop()
        while b[-1]==0:b.pop()
        if len(a)==len(b):
            if(a[-1]<b[-1]): a,b=b,a
        elif len(a)<len(b) : a,b=b,a
        b_=gcd(*b)
        a_=gcd(*a)
        b=[x//b_ for x in b]
        a=[x//a_ for x in a]
        while b:
            r = a.copy()
            
            while len(r) >= len(b):
                coeff = r[-1] // b[-1]
                for i in range(1,len(b)+1):
                    r[-i] -= coeff * b[-i]
                while r and r[-1]==0: r.pop()
            if not r: break
            
            r_=gcd(*r)
            r=[x//r_ for x in r]
            a, b = b, r
        
        b_=gcd(*b)
        b=[x//b_ for x in b]
        return QuotientPolynomialRing(b, poly1.pi_generator)
    
    @staticmethod
    def Inv(poly: 'QuotientPolynomialRing') -> 'QuotientPolynomialRing':
        mod=QuotientPolynomialRing(poly.pi_generator,poly.pi_generator)
        if QuotientPolynomialRing.GCD(poly,mod)!=1: raise Exception("Not invertible")
    
def aks_test(n):
    #FIRST TEST
    if is_perfect_power(n): return False

    #SECOND TEST
    l = n.bit_length() 
    m = 4*(l**2)
    r = 2
    while pair_gcd(r,n)==1:
        if pair_gcd(r, n) == 1:
            phi = euler_phi(r)
            order = phi
            for p, e in factor(phi):
                for _ in range(e):
                    if pow(n, order // p, r) == 1:
                        order //= p
                    else:
                        break
            if order > m:
                break
        r += 1

    #THIRD TEST
    if r==n: 
        return True

    #FOURTH TEST
    if pair_gcd(n,r)>1:
        return False

    #FIFTH TEST
    Mod=[-1]+[0]*(r-1)+[1]
    m=2*l*floor_sqrt(r)+1
    for j in range(1,m):
        LHS = QuotientPolynomialRing([j,1],Mod)
        RHS = QuotientPolynomialRing([j]+[0]*(n-1)+[1],Mod)
        L=LHS
        power=n
        L_ = QuotientPolynomialRing([1],Mod)
        while power!=0:
            if power % 2==1:
                L_ = QuotientPolynomialRing.Mul(L_,L)
            L = QuotientPolynomialRing.Mul(L,L)
            power//=2
        if L.element!=RHS.element: return False
    return True