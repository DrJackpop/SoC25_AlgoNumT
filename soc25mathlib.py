# Assignment 1
import typing
def pair_gcd(a :int ,b :int) -> int:
    """Function to calculate the GCD of 2 inputs
    
    Args:
        a (int): first parameter
        b (int): second parameter
    
    Returns:
        int: returns the greatest common divisor of the inputs.
    """
    while b!=0 : 
        a,b=b,a%b
    return int(a)

def pair_egcd(a: int, b: int) -> tuple[int, int, int]:
    """Funtion to calculate x,y such that xa+yb=gcd(a,b) and output x,y and the gcd, using the extended euclids elgorithm.

    Args:
        a (int): The first parameter
        b (int): The second parameter

    Returns:
        tuple[int, int, int]: returns rhe tuple x,y,d where xa+yb=d, where d is the greatest common divisor
    """
    if a%b==0 :
        return 0,1,b
    else : 
        x,y,d=pair_egcd(b,a%b)
        return y,x-a//b * y,d

def gcd(*args :int) -> int:
    """Function to calculate the GCD of a set of numbers

    Args:
        *args (int pointer): pointer to a integer list(same as array)
        
    Returns:
        int: returns the greatest common divisors of the numbers in the input
    """
    g=args[0]   
    for b in args:
        g=pair_gcd(g,b)
    return g

def pair_lcm(a: int, b: int) -> int:
    """Function to calculate the least common multiple of the inputs

    Args:
        a (int): The first parameter
        b (int): The second parameter

    Returns:
        int: returns the LCM of the 2 inputs
    """
    return int(a*b//gcd(a,b))

def lcm(*args : int) -> int:
    """Function to calculate the lcm of a set of numbers

    Args:
        *args (int pointer): pointer to a integer list(same as array)
        
    Returns:
        int: returns the least common multiple of the numbers in the input
    """
    l=args[0]
    for b in args:
        l=pair_lcm(l,b)
    return l

def are_relatively_prime(a : int,b :int) -> bool:
    """Function to determine if 2 numbers are relatively prime, ie if they have any common factors other than 1

    Args:
        a (int): The first parameter
        b (int): The second parameter

    Returns:
        bool: returns True is the numbers are relativelt prime, returns False if not relatively prime
    """
    if pair_gcd(a,b)==1 : 
        return True 
    return False

def mod_inv(a: int, n: int) -> int:
    """Function to return the modular inverse of a mod n

    Args:
        a (int): The numbers whose inverse we want to find
        n (int): The number which we are considering the modulo of

    Raises:
        Exception: 'a and n not comprime, no inverse modula exists', if there does not exist a modulo inverse of a mod n

    Returns:
        int: _description_
    """
    if pair_gcd(a,n)!=1 : raise Exception("a and n not coprime, no inverse modulo exists")
    x,y,d=pair_egcd(n,a)
    if x>0 : return n+y
    return y

def crt(a: list[int],n: list[int]) -> int:
    """Function to implement the Chinese Remainder Theorem.

    Args:
        a (list[int]): List that has the numbers that we want to find the congurence of
        n (list[int]): List that contains number whose we take modulo corresponding to the same index in a, we assume these are coprime

    Returns:
        int: returns a such that it satisfies all the congurgences given with respect to the two lists given
    """
    N,x=1,0
    for i in n : N*=i
    for i,j in enumerate(n) :
        x+=mod_inv(N//j,j)*N//j*a[i]
    return int(x%N)

def is_quadratic_residue_prime(a: int, p: int) -> int:
    """Function to determine whether a number is a quadratic residue prime, if there is a number modulo p such that its square is a

    Args:
        a (int): The number which we want to determine if it is a quadratic residue prime or now
        p (int): The prime for which we consider the modulo

    Returns:
        int: returns 1 if the number is a quadratic residue prime, -1 if it is not, and 0 if a and p are not coprime
    """
    x=pow(a,int((p-1)/2),p)
    if x>1 : return -1
    else : return x
    
def is_quadratic_residue_prime_power(a: int, p: int, e: int) -> int:
    """Function to determine whether a number is a quadratic residue prime to the power, if there is a number modulo p^e such that its square is a

    Args:
        a (int): The number which we want to determine if it is a quadratic residue prime or now
        p (int): The prime for which we consider the modulo
        e (int): The power of the prime we consider as the modulo

    Returns:
        int: returns 1 if the number is a quadratic residue prime to the power, -1 if it is not, and 0 if a and p^e are not coprime
    """
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

def floor_sqrt(x:int) -> int:
    """Function to get the greatest integer less than or equal to the square root of a number

    Args:
        x (int): The number whose square root we want

    Returns:
        int: returns the integer part of the square root of x
    """
    k = max((x.bit_length()-1)//2,0)
    m = 1<<k
    for i in range(k-1,-1,-1):
        if (m+(1<<i))**2 <=x : m+=1<<i
    return m


def floor_nth_root(a: int, b: int) -> int:
    """Function to get the greatest integer less than or equal to the nth root of a number

    Args:
        a (int): The number whose nth root we want to check
        b (int): The number of the exponent of the answer which approximates the first input

    Returns:
        int: returns the integer part of the nth root of a
    """
    k = (a.bit_length()-1)//b
    m = 1<<k
    for i in range(k-1,-1,-1):
        if (m +(1<<i))**b <= a:
            m += 1 << i
    return m
def is_perfect_power(x: int) -> bool:
    """Function to determine whether a number is a perfect power of another number
    
    Args:
        x (int): The number which we want to check

    Returns:
        bool: returns True if the number is a perfect power, False if the number is not a perfect power
    """
    l = x.bit_length()
    for b in range(2, l+1):
        a = floor_nth_root(x, b)
        if a**b == x or (a+1)**b == x: 
            return True
    return False

def is_prime(n: int) -> bool:
    """Function that implements the Miller-Rabin test to determine whether a number is prime or not using a probablistic test

    Args:
        n (int): the number which we want to check if it is prime or not

    Returns:
        bool: returns True if the input is prime(probably), False if input is not
    """
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

def gen_prime(m:int) -> int:
    """Function that generates a random prime between 2 and m(both inclusive)

    Args:
        m (int): The upper bound for which we want to get a prime

    Returns:
        int: returns a prime that is between 2 and m
    """
    p=m
    while not is_prime(p):
        p=random.randint(2,m)
    return p

def gen_k_bit_prime(k: int) -> int:
    """Function to generate a random prime of k bits

    Args:
        k (int): The number of bits we want our prime to be

    Returns:
        int: returns a k-bit prime
    """
    p: int=2**k-1
    while not is_prime(p):
        p=random.randint(2**(k-1),2**k)
        p|=1
    return p

def factor(n:int) -> list[tuple[int, int]]:
    """Function to calculate the prime factorisation of a number

    Args:
        n (int): The number whose prime factorisation we want to find

    Returns:
        list[tuple[int, int]]: returns the list of prime factors as a 2-tuple with the first element as the prime and second as the power to which the prime is raised in n
    """
    factor: list[tuple[int,int]]=[]
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

def euler_phi(n:int) -> int:
    """Function to calculate the Euler phi function, or the number of numbers less than n that are coprime with n

    Args:
        n (int): The first parameter

    Returns:
        int: returns the value of the Euler phi function of the first parameter
    """
    if n==1:return n
    fac=factor(n)
    euler=1
    for i,j in fac:
        euler*=(i**(j-1))*(i-1)
    return euler

class  QuotientPolynomialRing:
    """Class that is used to implement a Quotient Polynomial Ring over some integers modulo some monix polynomial of the ring.
    
    Attributes:
        poly (list[int]): the list that contains the coeffecients of the polynomial, the index is the power of x
        pi_gen (list[int]): the list that contains the monic polynomial over which the polynomial modulo is based
    """
    element: list[int]
    pi_gen: list[int]
    def __init__(self,poly:list[int],pi_gen:list[int]) -> None:
        """Function that initialises a object of the class QuotientPolynomialRing

        Args:
            poly (list[int]): The list that contains the coeffecients of the polynomial we consider
            pi_gen (list[int]): The monic polynomial over which the poly's modulo is considered
        """
        self.pi_generator=pi_gen
        self.element=self.modulo(poly)
    
    def modulo(self,poly:list[int]) -> list[int]:
        """Function that reduces a polynomial modulo another

        Args:
            poly (_type_): The polynomial we want to reduce

        Returns:
            QuotientPolynomialRing: returns the reduced polynomial over the polynomial ring with modulo pi_gen
        """
        if len(poly)<len(self.pi_generator): return poly+[0]*(len(self.pi_generator)-len(poly)-1)
        
        while len(poly)>len(self.pi_generator)-1:
            fac=poly[-1]
            for i in range(1,len(self.pi_generator)+1):
                poly[-i]=poly[-i]-fac*self.pi_generator[-i]
            poly.pop()
        return poly
    
    @staticmethod
    def Add(poly1: 'QuotientPolynomialRing', poly2: 'QuotientPolynomialRing') -> "QuotientPolynomialRing":
        """Function that adds two polynomials and reduces them over their common pi_gen

        Args:
            poly1 (QuotientPolynomialRing): The first polynomial 
            poly2 (QuotientPolynomialRing): The second polynomial

        Raises:
            ValueError: Raises error if the pi_gen of the two polynomials are different

        Returns:
            QuotientPolynomialRing: returns the sum of the polynomials reduces over the polynomial ring modulo pi_gen
        """
        poly1.element+=[0]*max(0,(len(poly1.pi_generator)-len(poly1.element)))
        poly2.element+=[0]*max(0,(len(poly2.pi_generator)-len(poly2.element)))
        if poly1.pi_generator != poly2.pi_generator:
            raise ValueError('Different pi_generators')
        add = []
        for _ in range(len(poly1.pi_generator)):
            add.append(poly1.element[_]+poly2.element[_])
        return QuotientPolynomialRing(add, poly1.pi_generator)

    @staticmethod
    def Sub(poly1: 'QuotientPolynomialRing', poly2: 'QuotientPolynomialRing') -> "QuotientPolynomialRing":
        """Function that subtracts two polynomials and reduces them over their common pi_gen

        Args:
            poly1 (QuotientPolynomialRing): The first polynomial 
            poly2 (QuotientPolynomialRing): The second polynomial

        Raises:
            ValueError: Raises error if the pi_gen of the two polynomials are different

        Returns:
            QuotientPolynomialRing: returns the difference of the polynomials reduces over the polynomial ring modulo pi_gen
        """
        
        poly1.element+=[0]*max(0,(len(poly1.pi_generator)-len(poly1.element)))
        poly2.element+=[0]*max(0,(len(poly2.pi_generator)-len(poly2.element)))
        if poly1.pi_generator != poly2.pi_generator:
            raise ValueError('Different pi_generators')
        sub = []
        for _ in range(len(poly1.pi_generator)):
            sub.append(poly1.element[_]-poly2.element[_])
        return QuotientPolynomialRing(sub, poly1.pi_generator)
    
    @staticmethod
    def Mul(poly1: 'QuotientPolynomialRing', poly2: 'QuotientPolynomialRing') -> "QuotientPolynomialRing":
        """Function that adds multiplies polynomials and reduces them over their common pi_gen

        Args:
            poly1 (QuotientPolynomialRing): The first polynomial 
            poly2 (QuotientPolynomialRing): The second polynomial

        Raises:
            ValueError: Raises error if the pi_gen of the two polynomials are different

        Returns:
            QuotientPolynomialRing: returns the multiplication of the polynomials reduces over the polynomial ring modulo pi_gen
        """
        if poly1.pi_generator != poly2.pi_generator:
            raise ValueError('Different pi_generators')
        p1=poly1.element
        p2=poly2.element
        while not p1[-1]: p1.pop()
        while not p2[-1]: p2.pop()
        p1l=len(p1)
        p2l=len(p2)    
        prod = [0] * (p2l+p1l-1) 
        for i in range(p1l):
            p=p1[i]
            if p==0:
                continue
            for j in range(p2l):
                if p2[j]==0: continue
                prod[i + j] += p * p2[j]
        return QuotientPolynomialRing(prod, poly1.pi_generator)
    
    @staticmethod
    def GCD(poly1: 'QuotientPolynomialRing', poly2: 'QuotientPolynomialRing') -> "QuotientPolynomialRing":
        """Function that finds the greatest common divisor that is a polynomial of two polynomials

        Args:
            poly1 (QuotientPolynomialRing): The first polynomial 
            poly2 (QuotientPolynomialRing): The second polynomial

        Raises:
            ValueError: Raises error if the pi_gen of the two polynomials are different

        Returns:
            QuotientPolynomialRing: returns the GCD of the polynomials over the polynomial ring modulo pi_gen
        """
        if poly1.pi_generator != poly2.pi_generator:
            raise ValueError('Different pi_generators')

        a = poly1.element.copy()
        b = poly2.element.copy()
        if len(a)==len(b):
            if(a[-1]<b[-1]): a,b=b,a
        elif len(a)<len(b) : a,b=b,a
        b_=gcd(*b)
        a_=gcd(*a)
        b=[int(x/b_) for x in b]
        a=[int(x/a_) for x in a]
        while not b[-1]:
            b.pop()
        while not a[-1]:
            a.pop()
        while b:
            flag=True
            if not flag: break
            r = a.copy()[::-1]
            b = b[::-1]
            while len(r) >= len(b):       
                if r[0] % b[0] != 0:
                    mul = b[0]
                    for i in range(len(r)):
                        r[i] *= mul
                coeff = r[0] // b[0]
                for i in range(0,len(b)):
                    r[i] -= coeff * b[i]
                while r and not r[0]: r.pop(0)
            
            a, b = b[::-1], r[::-1]
        a_=gcd(*a)
        a=[ai//a_ for ai in a]
        return QuotientPolynomialRing(a, poly1.pi_generator)
    
    @staticmethod
    def Inv(poly: 'QuotientPolynomialRing') -> "QuotientPolynomialRing":
        """Function that calculates the inverse polynomial over the polynomial ring modulo pi_gen

        Args:
            poly1 (QuotientPolynomialRing): The first polynomial

        Returns:
            QuotientPolynomialRing: returns modulo inverse of the polynomial over the polynomial ring modulo pi_gen
        """
        mod=poly.pi_generator  [::-1]
        po=poly.element[::-1]
        quotient: list[int]=[0]*(len(mod)-1)
        remainder: list[int]=[0]*(len(mod)-1)
        for i in range(len(mod)-len(po)):
            quotient[i]+=1
            
            
        
        return poly
        
    
def aks_test(n: int) -> bool:
    """Function that determines whether a number is prime, using a deterministic method

    Args:
        n (int): The number which want to see, if it is prime or now

    Returns:
        bool: returns True if the number is prime, False if it is not prime
    """
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
        print(m-j)
        L = QuotientPolynomialRing([j,1],Mod)
        rhs = [0]*r
        rhs[0] = j % n
        rhs[n%r] = (rhs[n%r] + 1) % n
        R = QuotientPolynomialRing(rhs, Mod)
        power=n
        L_ = QuotientPolynomialRing([1],Mod)
        while power!=0:
            if power % 2==1:
                L_ = QuotientPolynomialRing([l % n for l in QuotientPolynomialRing.Mul(L_, L).element],Mod)
            L = QuotientPolynomialRing([l % n for l in QuotientPolynomialRing.Mul(L, L).element],Mod)
            power//=2
        
        if L_.element!=R.element: return False
    return True

# Assignment 3

def get_generator(p : int) -> int:
    """Function that calculates a generator of the group (Z_p)*

    Args:
        p (int): The prime whose multiplicative integer modulo classes generator we want to find

    Returns:
        int: returns a generator of the multiplicative group (Z_p)*, returns 0 if no generator found (not possible)
    """
    l=factor(p-1)
    l=[(p-1//q,1) for (q,_) in l]
    g: int=0
    for g in range(1,p):
        for i,_ in l:
            if pow(g,i,p)==1: break
        else:
            return g
    return 0
    
def discrete_log(x:int, g:int,p:int) -> int:
    """Function that calculate the discrete log of a number of a certain base in the multiplicative group (Z_p)*

    Args:
        x (int): The number whose logarithm we want
        g (int): The base of the logarithm
        p (int): The prime of which the group is based on

    Returns:
        int: returns the discrete logarithm with the parameters as given
    """
    m = floor_sqrt(p-1) + 1
    T={}
    b=1
    for i in range(m):  
        T[b]=i
        b=(b * g) % p
    
    g_ = mod_inv(pow(g,m,p),p)
    b=x
    i=0
    while b not in T:
        b=(b*g_)%p
        i+=1
    return i*m+T[b]
def legendre_symbol(a: int,p: int) -> int:
    """Function that returns the value of the legendre symbol of two numbers a and p

    Args:
        a (int): The numerator in the legendre symbol
        p (int): The denominator in the lengendre symbol

    Returns:
        int: returns the value of the legendre symbol with the parameters given
    """
    if pair_gcd(a,p)>1: return 0
    return is_quadratic_residue_prime(a,p)

def jacobi_symbol(a: int,n: int) -> int:
    """Function that returns the value of the Jacobi symbol of two numbers a and n

    Args:
        a (int): The numerator in the Jacobi symbol
        n (int): The denominator in the Jacobi symbol

    Returns:
        int: returns the Jacobi symbol with the parameters provided
    """
    sig=1
    while True:
        a%=n
        if a==0:
            if n==1: return sig
            else: return 0
        h=0
        while a%2==0: 
            a//=2
            h+=1
        if h%2==1 and n%8!=1 and n%8!=7:
            sig*=-1
        if a%4==3 and n%4!=1:
            sig*=-1
        a,n=n,a
        
def modular_sqrt_prime(x: int,p: int) -> int:
    """Function that return the modular square root of a number in a given prime modulo

    Args:
        x (int): The number whose sqaure root we want to find
        p (int): The prime whose multiplicative modulo group we consider

    Raises:
        Exception: Raises exception if the square root of the number does not exist

    Returns:
        int: returns the square root of the given parameter with respect to the prime modulo group
    """
    l=legendre_symbol(x,p)
    if(l==-1): raise Exception("Exception: mod square root does not exist")
    if(l==0): return 0
    m=p-1
    while m%2==0:
        m//=2
    while True:
        g=random.randint(1,p-1)
        if pow(g, (p-1)//2,p)==p-1:
            break
    g=pow(g,m,p)
    y=discrete_log(pow(x,m,p),g,p)//2
    b=pow(g,y,p)*mod_inv(pow(x,(m//2),p),p)
    return b%p
        
def modular_sqrt_prime_power(x: int,p: int,e: int) -> int:
    """Function that returns the square root of a number in a given prime power modulo.

    Args:
        x (int): The number whose square root we want to find
        p (int): The prime whose powered we consider for the multiplicative modulo group
        e (int): The power of the prime that we take

    Returns:
        int: returns the square root of the given parameter with respect to the prime powered modulo group
    """
    b=modular_sqrt_prime(x,p)
    pj=p
    for _ in range(1,e):
        inv=mod_inv(2*b,p)
        t=(((x-b*b))//pj * inv)%p
        b+=t*pj
        b*=-1
        pj*=p
        b%=pj
    return b

def modular_sqrt(x: int,n: int) -> int:
    """Funciton that returns the square root of a number in a given modulo group.

    Args:
        x (int): The number whose square root we want to find
        n (int): The number whose multiplicative modulo group we want to consider

    Returns:
        int: returns the square root of the given parameter with respect to the provided numbers modulo group
    """
    f=factor(n)
    roots=[]
    for i,j in f:
        mod_sqrt=modular_sqrt_prime_power(x,i,j)
        roots.append(mod_sqrt)
    f=[(n//(m**i),m**i) for m,i in f]
    x=0
    for j,i in enumerate(roots):
        y=f[j]
        x+=i*y[0]*mod_inv(y[0],y[1])
    return min(x%n,n-x%n)

def is_smooth(m: int,y: int) -> bool:
    """Function that returns is a number is smooth with respect to another, if all prime factors of one parameter are less than the other parameter

    Args:
        m (int): The number which we want to check is smooth
        y (int): The number for which we check the other number is smooth

    Returns:
        bool: returns True if the number m is y-smooth, False if m is not y-smooth
    """
    if y%2==0: y+=1
    for i in range(y,floor_sqrt(m)+1,2):
        if not is_prime(i): continue
        if m%i==0: return False
    return True
def gausselim(A: list[list[int]], p: int) -> list[int]:
    """Function that uses gaussian elimination to calcualte the solution of a set of linear equations

    Args:
        M (list[list[int]]): The matrix that represents the linear equation system
        p (int): The modulo group of which we consider the number

    Returns:
        list[int]: The vector that is the solution of the relation Ax=b
    """
    m, n = len(A), len(A[0])
    M = [row.copy() for row in A]
    pivot_cols = []
    r = 0 

    for c in range(n):
        pivot = None
        for i in range(r, m):
            if M[i][c] % p != 0:
                pivot = i
                break
        if pivot is None:
            continue

        M[r], M[pivot] = M[pivot], M[r]

        inv = mod_inv(M[r][c],p)
        M[r] = [(val * inv) % p for val in M[r]]

        for i in range(m):
            if i == r:
                continue
            factor = M[i][c] % p
            if factor:
                M[i] = [(M[i][j] - factor * M[r][j]) % p for j in range(n)]

        pivot_cols.append(c)
        r += 1
        if r == m:
            break

    r = len(pivot_cols)
    if r == n:
        return [0]*n

    fc = [c for c in range(n) if c not in pivot_cols]
    x = [0] * n
    f = fc[0]
    x[f] = 1

    # Back-substitute: for each pivot row, x[pivot_col] = - sum_{j>pivot} M[row][j] * x[j]
    for i in reversed(range(r)):
        c = pivot_cols[i]
        s = 0
        for j in range(c + 1, n):
            s = (s + M[i][j] * x[j]) % p
        x[c] = (-s) % p

    return x
        
                    
def probabilistic_dlog(x: int,g: int,p: int) -> int:
    """Returns the discrete logarithm of a number with the base and the prime modulo we consider, using a probabilistic method

    Args:
        x (int): The number whose logarithm we want to find
        g (int): The base of the logarithm
        p (int): The prime whose modulo group we consider 

    Returns:
        int: returns the discrete logarithm of the number with the base and prime modulo(probably)
    """
    pw=p.bit_length()*((p.bit_length()-1).bit_length())**0.5
    y=min(int(pow(2.719,pw*0.49)),p-1)
    vectors: list[list[int]]=[]
    primes: list[int]=[]
    r_vector: list[int]=[]
    s_vector: list[int]=[]
    for i in range(2,y):
        if is_prime(i):
            primes.append(i)
    i=0
    while i!=len(primes)+1:
        i+=1
        while True:
            r=random.randint(0,p-2)
            s=random.randint(0,p-2)
            d=random.randint(1,p-1)
            b=(pow(g,r,p)*pow(x,s,p))%p
            m=(b*d)%p
            if is_smooth(m,y):
                e: list[int]=[]
                e_=0
                for j in primes:
                    m_=m
                    e.append(0)
                    while m_%j==0:
                        e[e_]+=1
                        m_//=j
                    e_+=1
                vectors.append(e)
                r_vector.append(r)
                s_vector.append(s)
                break
    q=p-2
    c: list[int]=gausselim(vectors,q)
    r=0
    s=0
    for _ in range(len(r_vector)):
        r=(r+r_vector[_]*c[_])%q
        s=(s+s_vector[_]*c[_])%q
    if s==0:
        return 0
    else: return (-r*mod_inv(s,q))%q
            
def probabilistic_factor(n: int) -> list[tuple[int,int]]:
    """Function that returns the prime factorisation of a given number, using a probabilistic method

    Args:
        n (int): The number whose prime factorisation we want to find

    Returns:
        list[tuple[int,int]]: returns a list of tuples with the first arguement of the tuple as the prime and the second as the power to which that prime is raised.
    """
    factor: list[tuple[int,int]]=[]
    if n == 1 or n==0:
        return factor
    p2=0
    if n%2==0:
        while n%2==0 :
            p2+=1
            n//=2
        factor.append((2,p2))
    if is_perfect_power(n):
        b=int(n.bit_length()*0.7)
        while True:
            a=floor_nth_root(n,b)
            if a**b==n:
                if is_prime(a):
                    factor.append((a,b))
                    return factor
            b-=1
            if b==1:
                break
    if is_prime(n):
        factor.append((n,1))
        return factor

    pw=n.bit_length()*((n.bit_length()-1).bit_length())**0.5
    y=min(int(pow(2.719,pw*0.49)),n-1)
    vectors: list[list[int]]=[]
    primes: list[int]=[]
    for i in range(2,y):
        if is_prime(i):
            primes.append(i)
    k=len(primes)
    
    i=0
    alpha: list[int]=[]
    while i!=k+2:
        i+=1
        while True:
            while True:
                a=random.randint(0,n-1)
                if pair_gcd(a,n)==1: 
                    break
            if i==1:
                while True:
                    d=random.randint(0,n-1)
                    if pair_gcd(d,n)==1: 
                        break
            m=(a**2 * d)%n
            if is_smooth(m,y):
                alpha.append(a)
                e: list[int]=[]
                e_=0
                for j in primes:
                    m_=m
                    e.append(0)
                    while m_%j==0:
                        e[e_]+=1
                        m_//=j
                    e_+=1
                e.append(1)
                vectors.append(e)
                break
    
    c=gausselim(vectors,2)
    a=1
    b=pow(d,e[-1]//2)
    for i in range(k+1):
        a*=alpha[i]**c[i]
        if i!=k:
            b*=primes[i]**(e[i]//2)
    g=a/b
    d=1
    if g%n==1 or g%n==n-1:
        0
    else:
        d=pair_gcd(g-1,n)
    if d==0:
        return factor
    factor=factor+probabilistic_factor(d)
    factor=factor+probabilistic_factor(n//d)
    for i in range(len(factor)):
        for j in range(i,len(factor)):
            if factor[j][0] < factor[i][0]:
                factor[j],factor[i]=factor[i],factor[j]
    i=0
    while True:
        if i>=len(factor)-1:
            break
        if factor[i][0]==factor[i+1][0]:
            factor[i]=(factor[i][0],factor[i][1]+factor[i+1][1])
            factor.pop(i+1)
        i+=1
    return factor
