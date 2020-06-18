import qlens
q = qlens.Lens()

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
    s2 = 0
    qs = 0.8
    theta = 30
    xc1 = 0.7
    xc2 = 0.7
    yc = 0.3

    q.lens_add(b=b, alpha=alpha, s=s1, q=qs, theta=theta, xc=xc1, yc=yc)
    q.lens_add(b=b, alpha=alpha, s=s2, q=qs, theta=theta, xc=xc2, yc=yc)
    q.lens_display()

tests = [
            test_lens_clear,
            test_lens_display,
            test_lens_add_fail,
            test_lens_add
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

test_lens_add()