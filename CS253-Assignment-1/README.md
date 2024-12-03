
# Assignment 1: Car Rental System in C++


The assignment implements a car rental system in C++ using object oriented programming paradigms. The code is present in [main.cpp](https://github.com/AdiJ101/CS253-Assi-1/blob/main/main.cpp). There are 3 kinds of users and each of them have different functionalities and specifications. They are outlined in the [Problem Statement](https://github.com/AdiJ101/CS253-Assi-1/blob/main/Problem%20statement-1.pdf).All functionalities of the statement have been implemented acccordingly. There are 3 txt files which serve as databases.

- [customer.txt](https://github.com/AdiJ101/CS253-Assi-1/blob/main/customer.txt) stores [id,password,name,fine dues, customer record] of customer.

- [employee.txt](https://github.com/AdiJ101/CS253-Assi-1/blob/main/employee.txt) stores [id,password,name,fine dues, employee record] of employee.

- [car.txt](https://github.com/AdiJ101/CS253-Assi-1/blob/main/car.txt) stores [id,model,Date of rent,user id of renter,car condition,car rent] of car.

NOTE: 
 A default manager exists in the systum with the following credentials:

- ID: 0
- Password: 0 

If no customer/employee exists in databases then new customer/employee is given customer record of 100.

Fine will increment by 5 per day if car returned later than return_date.

If customer/employee damages car then customer record is decreased by 10 till it reaches 0.

A customer is only rented a car upon request if it's customer/employee record is more than car condition. ( To ensure that cars in good condition are rented to customer/employee with better records).



## Deployment

To deploy this project run

```bash
  g++ main.cpp -o main
.\main
```
The system will run on the console. The data modified throughout the program run will be reflected in the txt files.

