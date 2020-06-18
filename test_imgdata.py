import qlens
q = qlens.Lens()

def test_img_load():
    q.imgdata_load("alphafit.dat") # Should succeed
    q.imgdata_display()

def test_img_write():
    q.imgdata_load("alphafit.dat")
    q.imgdata_write("sample.a") # Should be equal to the above

def test_img_clear():
    q.imgdata_clear()

def test_img_clear_fail():
    q.imgdata_clear(9, 0) # Expected Error: Runtime

def test_img_clear_ok():
    q.imgdata_clear(0, 1) # Should fail

def test_img_add_fail():
    q.imgdata_add() # Expected Error: Runtime

def test_img_add_ok():
    q.imgdata_clear()
    for i in range(0, 5):
        q.imgdata_add(0, i)
    q.imgdata_write("sample.a")

tests = [
            test_img_load, test_img_write, test_img_clear, 
            test_img_clear_fail, test_img_clear_ok,
            test_img_add_ok, test_img_add_fail
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