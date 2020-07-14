from qlens import *
q = QLens()

def test_lens_clear():
    q.lens_clear()

def test_lens_display():
    q.lens_display()

def test_lens_add_fail():
    q.lens_add()

def test_lens_add():
    # Adapted from alphafit.in
    b = 4.5
    alpha = 1
    s1 = 0
    s2 = 6
    qs = 0.8
    theta = 30
    xc1 = 0.7
    xc2 = 0.9
    yc = 0.3

    q.lens_add(b=b, alpha=alpha, s=s1, q=qs, theta=theta, xc=xc1, yc=yc)
    q.lens_add(b=b, alpha=alpha, s=s2, q=qs, theta=theta, xc=xc2, yc=yc)
    q.lens_display()

def test_lens_alpha_add():
    # Adapted from alphafit.in
    b = 4.5
    alpha = 1
    s1 = 0
    s2 = 6
    qs = 0.8
    theta = 30
    xc1 = 0.7
    xc2 = 0.9
    yc = 0.3
    z = 30

    q.lens_add_alpha(b=b, zl_in=z, alpha=alpha, s=s1, q=qs, theta=theta, xc=xc1, yc=yc)
    q.lens_add_alpha(b=b, zl_in=z, alpha=alpha, s=s2, q=qs, theta=theta, xc=xc2, yc=yc)
    q.lens_display()

def test_lens_shear_add():
    q.lens_add_shear(20, 0, 90, 90)
    q.lens_display()

def test_lens_shear_add_fail():
    q.lens_add_shear()

def test_update_lens():
    A = Alpha()
    A.update({"b": 20})

tests = [
            test_lens_clear,
            test_lens_display,
            test_lens_add_fail,
            test_lens_add,
            test_lens_alpha_add,
            test_lens_shear_add,
            test_lens_shear_add_fail,
            test_update_lens
        ]

for i in tests:
    try:
        i()
        print("[  OK  ] " + i.__name__)
    except Exception as e:
        if (i.__name__.endswith("fail")):  
            print("[  OK  ] " + i.__name__ + ": expected failure")
        else:
            print("[ FAIL ] Unexpected error for " + i.__name__ + ": " + str(e))