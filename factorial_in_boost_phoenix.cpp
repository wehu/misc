#include <boost/function.hpp>
#include <boost/phoenix.hpp>
#include <iostream>

using namespace boost::phoenix;
using namespace boost::phoenix::arg_names;
using namespace boost::phoenix::local_names;


int main()
{
    boost::function<unsigned(unsigned)> factorial =
        let(
            _a = construct<boost::function<unsigned(unsigned)> >()
        )[
            let(
                _b = lambda[
                         if_else(
                             arg1 > 1
                             , bind(ref(_a), arg1 - 1) * arg1
                             , 1
                         )
                     ]
            )[
                ref(_a) = _b
                , bind(ref(_a), arg1)
            ]
        ];
   std::cout << factorial(3) << std::endl;
   return 0;
}
