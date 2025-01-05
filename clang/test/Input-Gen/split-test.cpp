// RUN: inputgen-minimize %s

extern "C" {
    int fc(int a) {
        return a;
    }
    int fd(int a) {
        return a;
    }
}
namespace one {
    namespace two {
        int fa(int a) {
            return a;
        }
        int fb(int a) {
            return a;
        }
    }
}

using namespace one::two;
__attribute__((inputgen_entry))
int foo(int a, int b) {
    return fa(a) + fb(b) + fc(a) + fd(a);
}
