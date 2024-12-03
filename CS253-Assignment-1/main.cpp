// FileCppTutorialExample.cpp : This file contains the 'main' function. Program execution begins and ends there.
//

#include <iostream>
#include <fstream>
#include <string>
#include <limits>
#include <vector>



using namespace std;
int ID;

class User
{
private:
    /* data */
public:
    string name;
    int id;
    int password;
};






class Customer: public User
{
private:
    /* data */
public:
    static int customerCount;
    
    string rentedCars;
    int fineDue;
    // string degree;
    int customer_record;
   
    friend void addCustomer(void);
    friend void deleteData(void);
    void setCustomer(int i);
    void printCustomer(void);
    void readRentableCarData(void);
    void rentaCar(void);
    void rentedCarData(void);
    void returnRequest(void);
    void updateFine(int newFine);
    void updateCustomerRecord(void);
    // void addCustomer();
    // void print(Customer &obj);
};

class Employee: public User
{
private:
    /* data */
public:
    
    string rentedCars;
    int fineDue;
    // string degree;
    int customer_record;
   
    friend void addCustomer(void);
    friend void deleteData(void);
    void setEmployee(int i);
    void printEmployee(void);
    void readRentableCarData(void);
    void rentaCar(void);
    void rentedCarData(void);
    void returnRequest(void);
    void updateFine(int newFine);
    void updateEmployeeRecord(void);
   
    // void addCustomer();
    // void print(Customer &obj);
};

class Car
{
private:
    /* data */
public:

int id;
  string model;
  int currentCustomerId;
  int dueDateDay;
  int dueDateMonth;
  int dueDateYear;
  int condition;
  int rent;
//   friend void rentRequest(void);
//   friend void showDueDate(void);
  void setCarRentDate(int a, int b, int c, int d,int e);
};

int checkCarId(int id){
     Car car;
    ifstream read;
    read.open("car.txt");
    while (!read.eof()) {
        read >> car.id;
        
        read.ignore();
        getline(read, car.model)
;
        
        // getline(read, customer.name);
        read >> car.currentCustomerId;
        // read >> customer.degree;
        read >> car.dueDateDay;
        read >> car.dueDateMonth;
        read >> car.dueDateYear;
        read >> car.condition;
        read >> car.rent;
        if (car.id == id) {
            // cout<<"\n\t"<<endl;
            return 1;
        }
    }
    read.close();
    return 0;   
}

int checkCarId2(int returnid, int id){
    Car car;
    ifstream read;
    read.open("car.txt");
    while (!read.eof()) {
        read >> car.id;
        
        read.ignore();
        getline(read, car.model)
;
        
        // getline(read, customer.name);
        
        // read >> customer.degree;
        read >> car.dueDateDay;
        read >> car.dueDateMonth;
        read >> car.dueDateYear;
        read >> car.currentCustomerId;
        read >> car.condition;
        read >> car.rent;
         //cout<<car.id<<"\t"<<returnid<<"\t"<<car.currentCustomerId<<"\t"<<id<<endl;
        if (car.id == returnid && car.currentCustomerId==id) {
           // cout<<car.id<<"\t"<<returnid<<"\t"<<car.currentCustomerId<<"\t"<<id<<endl;
            return 1;
        }
    }
    read.close();
    return 0;  
}

void addCar() {
    Car car;
    cout << "\n\tEnter car ID : ";
     cin>>car.id;;
         
while(std::cin.fail() || car.id<1 || checkCarId(car.id)) {
    std::cin.clear();
    std::cin.ignore(std::numeric_limits<std::streamsize>::max(),'\n');
    if(checkCarId(car.id)){std::cout<<"\n\tCar ID already exists \n\tEnter new ID: ";}
    else{
        std::cout << "\n\tInvaild ID.  Enter a (+) DIGITS : ";
    }
    std::cin >> car.id;;
}
    cout << "\n\tEnter car model name (single word): ";
    cin>>car.model;
    // getline(cin, ); //Nouman Habib
    cout << "\n\tRate the car condition (0- 100) : ";
    cin >> car.condition;
    while(std::cin.fail() || car.condition<0 || car.condition>100) {
    std::cin.clear();
    std::cin.ignore(std::numeric_limits<std::streamsize>::max(),'\n');
    std::cout << "\n\tInvaild entry, Please enter (+)Number (0- 100)";
        cout<<"\n\t";
    std::cin >> car.condition;
    }

    cout << "\n\tEnter car rent : ";
    cin >> car.rent;
    while(std::cin.fail() || car.rent<0 || car.rent>100000) {
    std::cin.clear();
    std::cin.ignore(std::numeric_limits<std::streamsize>::max(),'\n');
    
        std::cout << "\n\tInvaild entry, Please enter (+)Number";
        cout<<"\n\t";
        std::cin >> car.rent;
    }
        
    car.dueDateDay=-1; 
    car.dueDateMonth=-1;
    car.dueDateYear=-1;  
    car.currentCustomerId=-1;
    
    // ID++;

    ofstream write;
    write.open("car.txt", ios::app);
    // write << "\n" << ID;
    write << "\n" << car.id ;
    write << "\n" << car.model ;
    write << "\n" << car.currentCustomerId ;
    write << "\n" << car.dueDateDay ;
    write << "\n" << car.dueDateMonth ;
    write << "\n" << car.dueDateYear ;
    write << "\n" << car.condition;
    write << "\n" << car.rent ;
    write.close();
    // write.open("id.txt");
    // write << ID;
    // write.close();
    cout << "\n\tData save to file";
}

void printCar(Car s) {
    cout << "\n\t---Car Data---";
    cout << "\n\tCar ID : " << s.id;
    cout<<"\n\tModel : "<<s.model;
    if(s.currentCustomerId==-1){
        cout<<"\n\tAvailable";
    }
    else{
        cout << "\n\tCar rented to : " << s.currentCustomerId;
        cout << "\n\tRented on : " << s.dueDateDay<<"/"<<s.dueDateMonth<<"/"<<s.dueDateYear;
    }
    cout << "\n\tCar condition : " << s.condition;
    // cout << "\n\tDegree : " << s.degree;
    cout << "\n\tCar rent : " << s.rent<<endl;

}

void readAvailableCarData(){
    Car car;
    int numberofCars=0;
    ifstream read;
    read.open("car.txt");
    while (!read.eof()) {
        read >> car.id;
        read.ignore();
        getline(read, car.model);
        read >> car.dueDateDay;
        read >> car.dueDateMonth;
        read >> car.dueDateYear;
        read >> car.currentCustomerId;
        read >> car.condition;
        read >> car.rent;
        if(car.currentCustomerId==-1){printCar(car);
        numberofCars++;
        }
    }
    read.close();
    cout<<"\n\tTotal number of available cars : "<<numberofCars;
}

void readAvailableCarDataEmployee(){
        Car car;
    int numberofCars=0;
    ifstream read;
    read.open("car.txt");
    while (!read.eof()) {
        read >> car.id;
        read.ignore();
        getline(read, car.model);
        read >> car.dueDateDay;
        read >> car.dueDateMonth;
        read >> car.dueDateYear;
        read >> car.currentCustomerId;
        read >> car.condition;
        read >> car.rent;
        int employeeRent= (((85*car.rent)/100));
        car.rent=employeeRent;
        if(car.currentCustomerId==-1){printCar(car);
        numberofCars++;
        }
    }
    read.close();
    cout<<"\n\tTotal number of available cars : "<<numberofCars;
}

void readCarData() {
    Car car;
    int numberofCars=0;
    ifstream read;
    read.open("car.txt");
    while (!read.eof()) {
        read >> car.id;
        read.ignore();
        getline(read, car.model);
        read >> car.dueDateDay;
        read >> car.dueDateMonth;
        read >> car.dueDateYear;
        read >> car.currentCustomerId;
        read >> car.condition;
        read >> car.rent;
        printCar(car);
        numberofCars++;
    }
    read.close();
    cout<<"\n\tTotal number of cars : "<<numberofCars;
}

int searchCarData() {
    int id;
    cout << "\n\tEnter car id want to search : ";
    cin >> id;
    while(std::cin.fail() || id<1) {
    std::cin.clear();
    std::cin.ignore(std::numeric_limits<std::streamsize>::max(),'\n');
        std::cout << "\n\tInvaild ID.  Enter a (+) DIGITS : ";
    std::cin >> id;;
}
if(!checkCarId(id)){std::cout<<"\n\tNo such car ID exists";
return 0;}
    Car car;
    ifstream read;
    read.open("car.txt");
    while (!read.eof()) {

         read >> car.id;
        read.ignore();
        getline(read, car.model);
        read >> car.dueDateDay;
        read >> car.dueDateMonth;
        read >> car.dueDateYear;
        read >> car.currentCustomerId;
        read >> car.condition;
        read >> car.rent;

        // read >> student.id;
        // read.ignore();
        // getline(read, student.name);
        // read >> student.age;
        // read >> student.degree;
        // read >> student.semesterNo;
        if (car.id == id) {
            printCar(car);
            return id;
        }
    }
    
       throw "No such car exists!";
}

void deleteCarData() {

    int id;
    
    try{
        id = searchCarData();
        
    } catch(const char* msg){
        throw "\n\tNo such car ID found";
        cerr<<msg<<endl;
        
    }
    cout << "\n\tDelete record (y/n) : ";
    char choice;
    cin >> choice;
    while(std::cin.fail() || (choice!='Y' && choice!='y'&& choice!='n' && choice!='N')) {
    std::cin.clear();
    std::cin.ignore(std::numeric_limits<std::streamsize>::max(),'\n');
    std::cout << "\n\tInvaild input, Please enter (y/n) : ";
        cout<<"\n\t";
    std::cin >> choice;
    }

    // int id = searchData();
    // cout << "\n\tYou want to delete record (y/n) : ";
    // char choice;
    // cin >> choice;
    if (choice == 'y') {
        Car car;
        ofstream tempFile;
        tempFile.open("temp.txt");
        ifstream read;
        read.open("car.txt");
        while (!read.eof()) {
            read >> car.id;
            read.ignore();
            getline(read, car.model);
            read >> car.dueDateDay;
            read >> car.dueDateMonth;
            read >> car.dueDateYear;
            read >> car.currentCustomerId;
            read >> car.condition;
            read >> car.rent;
            if (car.id != id) {
                tempFile << "\n" << car.id;
                tempFile << "\n" << car.model;
                tempFile << "\n" << car.dueDateDay;
                tempFile << "\n" << car.dueDateMonth;
                tempFile << "\n" << car.dueDateYear;
                tempFile << "\n" << car.currentCustomerId;
                tempFile << "\n" << car.condition;
                tempFile << "\n" << car.rent;
            }
        }
        read.close();
        tempFile.close();
        remove("car.txt");
        rename("temp.txt", "car.txt");
        cout << "\n\tCar data deleted successfuly";
    }
    else {
        cout << "\n\tRecord not deleted";
    }
}


void Car::setCarRentDate(int a, int b, int c, int d,int e){
        Car car;
        ofstream tempFile;
        tempFile.open("temp.txt");
        ifstream read;
        read.open("car.txt");
        while (!read.eof()) {
            read >> car.id;
            read.ignore();
            getline(read, car.model);
            read >> car.dueDateDay;
            read >> car.dueDateMonth;
            read >> car.dueDateYear;
            // read >> customer.degree;
            read >> car.currentCustomerId;
            read >> car.condition;
            read >> car.rent;
            if (car.id != id) {
                tempFile << "\n" << car.id;
                tempFile << "\n" << car.model;
                tempFile << "\n" << car.dueDateDay;
                tempFile << "\n" << car.dueDateMonth;
                tempFile << "\n" << car.dueDateYear;
                tempFile << "\n" << car.currentCustomerId;
                tempFile << "\n" << car.condition;
                tempFile << "\n" << car.rent;
            }
            else {
                    tempFile << "\n"<< car.id;
                tempFile << "\n" << car.model;
                if(e==1){
                    tempFile << "\n" << a;
                tempFile << "\n" << b;
                tempFile << "\n" << c;
                tempFile << "\n" << d;
                }
                else{
                     tempFile << "\n" << -1;
                tempFile << "\n" << -1;
                tempFile << "\n" << -1;
                tempFile << "\n" << -1;
                }
                tempFile << "\n" << car.condition;
                tempFile << "\n" << car.rent;
            }
        }
        read.close();
        tempFile.close();
        remove("car.txt");
        rename("temp.txt", "car.txt");
        cout << "\n\tData updated successfuly";
     
     
}

void updateCarData() {
    int id;
    try{
        id = searchCarData();
    } catch(const char* msg){
                cerr<<msg<<endl;
                return;
    }
    
    if(checkCarId(id)){cout << "\n\tYou want to update record (y/n) : ";
    char choice;
    int option;
    cin >> choice;
    while(std::cin.fail() || (choice!='Y' && choice!='y'&& choice!='n' && choice!='N')) {
    std::cin.clear();
    std::cin.ignore(std::numeric_limits<std::streamsize>::max(),'\n');
    std::cout << "\n\tInvaild input, Please enter (y/n) : ";
        cout<<"\n\t";
    std::cin >> choice;
    }
//      if(cin.fail()){
//             cin.ignore();
//             throw "\n\tInvalid input";
// }
    if (choice == 'y' || choice =='Y') {
        Car newData;
        cout<<"\n\tWhat do you will like to update : ";
        cout<<"\n\t1. Car ID";
        cout<<"\n\t2. Model";
        cout<<"\n\t3. Car condition";
        cout<<"\n\t4. Car rent";
        cout<<"\n\tEnter choice : ";
        cin>>option;
        while(std::cin.fail() || option<1 || option>4) {
    std::cin.clear();
    std::cin.ignore(std::numeric_limits<std::streamsize>::max(),'\n');
    std::cout << "\n\tInvaild entry, choose one out of 4 choices";
    //  cout<<"\n\tWhat do you will like to update : ";
        cout<<"\n\tWhat do you will like to update : ";
        cout<<"\n\t1. Car ID";
        cout<<"\n\t2. Model";
        cout<<"\n\t3. Car condition";
        cout<<"\n\t4. Car rent";
        cout<<"\n\tEnter choice : ";
    std::cin >> option;
    }
    
   
//         if(cin.fail()){
//             cin.ignore();
//             throw "\n\tInvalid input";
// }
           if(option==1){
            cout << "\n\tEnter new car ID : ";
            cin >> newData.id;
            while(std::cin.fail() || newData.id<1 || checkCarId(newData.id)) {
            std::cin.clear();
            std::cin.ignore(std::numeric_limits<std::streamsize>::max(),'\n');
            if(checkCarId(newData.id)){std::cout<<"\n\tCar ID already exists \n\tEnter new car ID: ";
            }
            else{
                std::cout << "\n\tInvaild ID.  Enter a (+) DIGITS : ";
            }
                std::cin >> newData.id;;
        }
        }

         else if(option==2){
            cout << "\n\tEnter new car model : ";
        cin>>newData.model;
        if(cin.fail()){
            cin.ignore();
            throw "\n\tInvalid input";
}
        }
        
        else if(option==3){
            cout << "\n\tEnter new car condition : ";
        cin >> newData.condition;
         while(std::cin.fail() || newData.condition<0 || newData.condition>100) {
    std::cin.clear();
    std::cin.ignore(std::numeric_limits<std::streamsize>::max(),'\n');
    std::cout << "\n\tInvaild condition data\n\tPlease enter within range (0- 100) : ";
        // cout<<"\n\t";
    std::cin >> newData.condition;
    }
        }
        
        // cout << "\n\tEnter customer degree : ";
        // cin >> newData.degree;
        else if(option==4){
            cout << "\n\tEnter new car rent : ";
         cin >> newData.rent;
         while(std::cin.fail() || newData.rent<0) {
    std::cin.clear();
    std::cin.ignore(std::numeric_limits<std::streamsize>::max(),'\n');
    std::cout << "\n\tInvaild rent, Please enter (+)Number : ";
        // cout<<"\n\t";
    std::cin >> newData.rent;
    }
        }
        
         else{
        //     cout<<"\tInvalid input\n";
        throw "\n\tInvalid Input";
        //     return;
         }
        Car car;
        ofstream tempFile;
        tempFile.open("temp.txt");
        ifstream read;
        read.open("car.txt");
        while (!read.eof()) {
            read >> car.id;
            read.ignore();
            getline(read, car.model);
            read >> car.dueDateDay;
            read >> car.dueDateMonth;
            read >> car.dueDateYear;
            // read >> customer.degree;
            read >> car.currentCustomerId;
            read >> car.condition;
            read >> car.rent;
            if (car.id != id) {
                tempFile << "\n" << car.id;
                tempFile << "\n" << car.model;
                tempFile << "\n" << car.dueDateDay;
                tempFile << "\n" << car.dueDateMonth;
                tempFile << "\n" << car.dueDateYear;
                tempFile << "\n" << car.currentCustomerId;
                tempFile << "\n" << car.condition;
                tempFile << "\n" << car.rent;
            }
            else {
                
                if(option==1){
                    tempFile << "\n"<< newData.id;
                tempFile << "\n" << car.model;
                tempFile << "\n" << car.dueDateDay;
                tempFile << "\n" << car.dueDateMonth;
                tempFile << "\n" << car.dueDateYear;
                tempFile << "\n" << car.currentCustomerId;
                tempFile << "\n" << car.condition;
                tempFile << "\n" << car.rent;
            
                }        
                else if(option==2){
                    tempFile << "\n"<< car.id;
                tempFile << "\n" << newData.model;
                tempFile << "\n" << car.dueDateDay;
                tempFile << "\n" << car.dueDateMonth;
                tempFile << "\n" << car.dueDateYear;
                tempFile << "\n" << car.currentCustomerId;
                tempFile << "\n" << car.condition;
                tempFile << "\n" << car.rent;
                }
                // tempFile << "\n" << newData.degree;
                else if(option==3){
                    tempFile << "\n"<< car.id;
                tempFile << "\n" << car.model;
                tempFile << "\n" << car.dueDateDay;
                tempFile << "\n" << car.dueDateMonth;
                tempFile << "\n" << car.dueDateYear;
                tempFile << "\n" << car.currentCustomerId;
                tempFile << "\n" << newData.condition;
                tempFile << "\n" << car.rent;
                }
                else if(option==4){
                    tempFile << "\n"<< car.id;
                tempFile << "\n" << car.model;
                tempFile << "\n" << car.dueDateDay;
                tempFile << "\n" << car.dueDateMonth;
                tempFile << "\n" << car.dueDateYear;
                tempFile << "\n" << car.currentCustomerId;
                tempFile << "\n" << car.condition;
                tempFile << "\n" << newData.rent;
                }
            }
        }
        read.close();
        tempFile.close();
        remove("car.txt");
        rename("temp.txt", "car.txt");
        cout << "\n\tData updated successfuly";
    }
    else  if (choice == 'n' || choice =='N') {
        cout<<"\n\tNo change implemented";
        
    }
    else{
        throw "\n\tRecord not updated";
    }
    }
}



int checkUser(int id){
     Customer customer;
    ifstream read;
    read.open("customer.txt");
    while (!read.eof()) {
        read >> customer.id;
        read >> customer.password;
        read.ignore();
        getline(read, customer.rentedCars)
;
        
        getline(read, customer.name);
        read >> customer.fineDue;
        // read >> customer.degree;
        read >> customer.customer_record;
        // cout<<customer.id<<endl;
        if (customer.id == id) {
            // cout<<"\n\t"<<endl;
            return 1;
        }
    }
    read.close();
     
    Employee customer3;
    ifstream read2;
    read2.open("employee.txt");
    while (!read2.eof()) {
        read2 >> customer3.id;
        read2 >> customer3.password;
        read2.ignore();
        getline(read2, customer3.rentedCars)
;
        
        getline(read2, customer3.name);
        read2 >> customer3.fineDue;
        // read >> customer.degree;
        read2 >> customer3.customer_record;
        //cout<<customer3.id<<endl;
        if (customer3.id == id) {
            // cout<<"\n\tcheckusertest"<<endl;
            return 1;
        }
    }
    read2.close();
    return 0;   
}

int validateCustomer(int id, int password){
     Customer customer;
    ifstream read;
    read.open("customer.txt");
    while (!read.eof()) {
        read >> customer.id;
        read >> customer.password;
        read.ignore();
        getline(read, customer.rentedCars)
;
        
        getline(read, customer.name);
        read >> customer.fineDue;
        // read >> customer.degree;
        read >> customer.customer_record;
        if ((customer.id == id) && (customer.password==password)) {
            // cout<<"\n\t"<<endl;
            return 1;
        }
    }
    read.close();
     
    return 0;  
}

int validateEmployee(int id, int password){
    Employee customer;
    ifstream read;
    read.open("employee.txt");
    while (!read.eof()) {
        read >> customer.id;
        read >> customer.password;
        read.ignore();
        getline(read, customer.rentedCars)
;
        
        getline(read, customer.name);
        read >> customer.fineDue;
        // read >> customer.degree;
        read >> customer.customer_record;
        if ((customer.id == id) && (customer.password==password)) {
            // cout<<"\n\t"<<endl;
            return 1;
        }
    }
    read.close();
}

void Customer::printCustomer() {
    cout << "\n\t---Customer Data---";
    cout << "\n\tID : " << id;
    cout<<"\n\tName : "<<name;
    //<< rentedCars;
    cout << "\n\tFine due : " << fineDue;
    // cout << "\n\tDegree : " << s.degree;
    cout << "\n\tCustomer Record : " << customer_record;
    cout<<"\n\tCurrently rented cars : \n";
        this->rentedCarData();
}

void Employee::printEmployee() {
    cout << "\n\t---Employee Data---";
    cout << "\n\tID : " << id;
    cout<<"\n\tName : "<<name;
    //<< rentedCars;
    cout << "\n\tFine due : " << fineDue;
    // cout << "\n\tDegree : " << s.degree;
    cout << "\n\tCustomer Record : " << customer_record;
    cout<<"\n\tCurrently rented cars : \n";
        this->rentedCarData();
}

void Customer::updateFine(int newFine){
    int option=2;
     Customer newData;

        
           if(option==2){
            
        newData.fineDue=fineDue+newFine;
        }

        Customer customer;
        ofstream tempFile;
        tempFile.open("temp.txt");
        ifstream read;
        read.open("customer.txt");
        while (!read.eof()) {
            read >> customer.id;
            read >> customer.password;

            read.ignore();
            getline(read, customer.rentedCars);
            getline(read, customer.name);

            read >> customer.fineDue;
            // read >> customer.degree;
            read >> customer.customer_record;
            if (customer.id != id) {
                tempFile << "\n" << customer.id;
                tempFile << "\n" << customer.password;
                tempFile << "\n" << customer.rentedCars;
        tempFile << "\n" << customer.name;
                tempFile << "\n" << customer.fineDue;
                // tempFile << "\n" << customer.degree;
                tempFile << "\n" << customer.customer_record;
            }
            else {
                if(option==2){
                    tempFile << "\n"<< customer.id;
                    tempFile << "\n" << customer.password;
                    tempFile << "\n" << customer.rentedCars;
                    tempFile << "\n" << customer.name;
                    tempFile << "\n"<< newData.fineDue;
                    tempFile << "\n" << customer.customer_record;
                }
                // tempFile << "\n" << newData.degree;
                
                
            }
        }
        read.close();
        tempFile.close();
        remove("customer.txt");
        rename("temp.txt", "customer.txt");
        cout << "\n\tData updated successfuly";
}

void Employee::updateFine(int newFine){
    int option=2;
     Employee newData;

        
           if(option==2){
            
        newData.fineDue=fineDue+newFine;
        }

        Employee customer;
        ofstream tempFile;
        tempFile.open("temp.txt");
        ifstream read;
        read.open("employee.txt");
        while (!read.eof()) {
            read >> customer.id;
            read >> customer.password;

            read.ignore();
            getline(read, customer.rentedCars);
            getline(read, customer.name);

            read >> customer.fineDue;
            // read >> customer.degree;
            read >> customer.customer_record;
            if (customer.id != id) {
                tempFile << "\n" << customer.id;
                tempFile << "\n" << customer.password;
                tempFile << "\n" << customer.rentedCars;
        tempFile << "\n" << customer.name;
                tempFile << "\n" << customer.fineDue;
                // tempFile << "\n" << customer.degree;
                tempFile << "\n" << customer.customer_record;
            }
            else {
                if(option==2){
                    tempFile << "\n"<< customer.id;
                    tempFile << "\n" << customer.password;
                    tempFile << "\n" << customer.rentedCars;
                    tempFile << "\n" << customer.name;
                    tempFile << "\n"<< newData.fineDue;
                    tempFile << "\n" << customer.customer_record;
                }
                // tempFile << "\n" << newData.degree;
                
                
            }
        }
        read.close();
        tempFile.close();
        remove("employee.txt");
        rename("temp.txt", "employee.txt");
        cout << "\n\tData updated successfuly";
}

// void Customer::print(){
//     cout << "\n\t---Customer Data---";
//     cout << "\n\tID : " << id;
//     cout<<"\n\tName : "<<name;
//     //<< rentedCars;
//     cout << "\n\tFine due : " << fineDue;
//     // cout << "\n\tDegree : " << s.degree;
//     cout << "\n\tCustomer Record : " << customer_record;
//     cout<<"\n\tCurrently rented cars : \n";
//         this->rentedCarData();
// }

void Customer::updateCustomerRecord(){
    
    int option=2;
     Customer newData;

        
           if(option==2){
            if(customer_record>=10){
                newData.customer_record=customer_record-10;
            }
            else{
                newData.customer_record=0;
            }
        }

        Customer customer;
        ofstream tempFile;
        tempFile.open("temp.txt");
        ifstream read;
        read.open("customer.txt");
        while (!read.eof()) {
            read >> customer.id;
            read >> customer.password;

            read.ignore();
            getline(read, customer.rentedCars);
            getline(read, customer.name);

            read >> customer.fineDue;
            // read >> customer.degree;
            read >> customer.customer_record;
            if (customer.id != id) {
                tempFile << "\n" << customer.id;
                tempFile << "\n" << customer.password;
                tempFile << "\n" << customer.rentedCars;
        tempFile << "\n" << customer.name;
                tempFile << "\n" << customer.fineDue;
                // tempFile << "\n" << customer.degree;
                tempFile << "\n" << customer.customer_record;
            }
            else {
                if(option==2){
                    tempFile << "\n"<< customer.id;
                    tempFile << "\n" << customer.password;
                    tempFile << "\n" << customer.rentedCars;
                    tempFile << "\n" << customer.name;
                    tempFile << "\n"<< customer.fineDue;
                    tempFile << "\n" << newData.customer_record;
                }
                // tempFile << "\n" << newData.degree;
                
                
            }
        }
        read.close();
        tempFile.close();
        remove("customer.txt");
        rename("temp.txt", "customer.txt");
}

void Employee::updateEmployeeRecord(){
    
    int option=2;
     Employee newData;

        
           if(option==2){
            if(customer_record>=10){
                newData.customer_record=customer_record-10;
            }
            else{
                newData.customer_record=0;
            }
        }

        Customer customer;
        ofstream tempFile;
        tempFile.open("temp.txt");
        ifstream read;
        read.open("employee.txt");
        while (!read.eof()) {
            read >> customer.id;
            read >> customer.password;

            read.ignore();
            getline(read, customer.rentedCars);
            getline(read, customer.name);

            read >> customer.fineDue;
            // read >> customer.degree;
            read >> customer.customer_record;
            if (customer.id != id) {
                tempFile << "\n" << customer.id;
                tempFile << "\n" << customer.password;
                tempFile << "\n" << customer.rentedCars;
        tempFile << "\n" << customer.name;
                tempFile << "\n" << customer.fineDue;
                // tempFile << "\n" << customer.degree;
                tempFile << "\n" << customer.customer_record;
            }
            else {
                if(option==2){
                    tempFile << "\n"<< customer.id;
                    tempFile << "\n" << customer.password;
                    tempFile << "\n" << customer.rentedCars;
                    tempFile << "\n" << customer.name;
                    tempFile << "\n"<< customer.fineDue;
                    tempFile << "\n" << newData.customer_record;
                }
                // tempFile << "\n" << newData.degree;
                
                
            }
        }
        read.close();
        tempFile.close();
        remove("employee.txt");
        rename("temp.txt", "employee.txt");
}


void Customer::readRentableCarData(){
    Car car;
    int numberofCars=0;
    ifstream read;
    read.open("car.txt");
    while (!read.eof()) {
        read >> car.id;
        read.ignore();
        getline(read, car.model);
        read >> car.dueDateDay;
        read >> car.dueDateMonth;
        read >> car.dueDateYear;
        read >> car.currentCustomerId;
        read >> car.condition;
        read >> car.rent;
        if((customer_record>=car.condition) && (car.currentCustomerId==-1)){printCar(car);
        numberofCars++;}
        
    }
    read.close();
    cout<<"\n\tTotal number of rentable cars : "<<numberofCars;
}


void Employee::readRentableCarData(){
    Car car;
    int numberofCars=0;
    ifstream read;
    read.open("car.txt");
    while (!read.eof()) {
        read >> car.id;
        read.ignore();
        getline(read, car.model);
        read >> car.dueDateDay;
        read >> car.dueDateMonth;
        read >> car.dueDateYear;
        read >> car.currentCustomerId;
        read >> car.condition;
        read >> car.rent;
        int employeeRent= (((85*car.rent)/100));
        car.rent=employeeRent;
        if((customer_record>=car.condition) && (car.currentCustomerId==-1)){printCar(car);
        numberofCars++;}
        
    }
    read.close();
    cout<<"\n\tTotal number of rentable cars : "<<numberofCars;
}


int dateValidator(int a , int b, int c){
    if((a>=1 && a<=31) && (b==1 || b==3 || b==5 || b==7 || b==8 || b==10 || b==12))
                return 1;
            else if((a>=1 && a<=30) && (b==4 || b==6 || b==9 || b==11))
                return 1;
            else if((a>=1 && a<=28) && (b==2))
                return 1;
            else if(a==29 && b==2 && (c%400==0 ||(c%4==0 && c%100!=0)))
                return 1;
            else{
                return 0;
			}
}


void Customer::rentaCar(){
    int carId;
    cout << "\n\tEnter car id want to rent : ";
    cin >> carId;
    while(std::cin.fail() || carId<1) {
    std::cin.clear();
    std::cin.ignore(std::numeric_limits<std::streamsize>::max(),'\n');
        std::cout << "\n\tInvaild ID.  Enter a (+) DIGITS : ";
    std::cin >> carId;;
}
if(!checkCarId(carId)){std::cout<<"\n\tNo such car ID exists";
return ;
}
    Car car;
    ifstream read;
    read.open("car.txt");
    while (!read.eof()) {

         read >> car.id;
        read.ignore();
        getline(read, car.model);
        read >> car.dueDateDay;
        read >> car.dueDateMonth;
        read >> car.dueDateYear;
        read >> car.currentCustomerId;
        read >> car.condition;
        read >> car.rent;

        if (car.id == carId) {
            read.close();
            if(customer_record<car.condition){
                cout<<"\n\tLow customer record"<<"\n\t"<<"Need to increase customer record by : "<<car.condition-customer_record<<"\n\tYou cannot rent this car!";
                return;
            }
            else if(car.currentCustomerId>-1){
                cout<<"\n\tAlready rented";
                return;
            }
            else{
                cout<<"\n\tYou can rent "<<car.model;
            cout<<"\n\tCar rent is : "<<car.rent;
            cout<<"\n\tNOTE : Fine of Rs.5 per day if car not returned within 30 days";
            cout<<"\n\tRent car"<<car.model<< "\t(y/n)";
            cout<<"\n\tEnter choice : ";
            char choice;
    cin >> choice;
    while(std::cin.fail() || (choice!='Y' && choice!='y'&& choice!='n' && choice!='N')) {
    std::cin.clear();
    std::cin.ignore(std::numeric_limits<std::streamsize>::max(),'\n');
    std::cout << "\n\tInvaild input, Please enter (y/n) : ";
        cout<<"\n\t";
    std::cin >> choice;
    }
//      if(cin.fail()){
//             cin.ignore();
//             throw "\n\tInvalid input";
// }
    if (choice == 'y' || choice =='Y') {
        cout<<"\n\tEnter current date\n\tRange (01/01/1990) to (31/12/9999)";
        int a,b,c;
        cout<<"\n\tEnter Year : ";
        cin>>c;
        while(std::cin.fail() || c<1990 || c>9999) {
    std::cin.clear();
    std::cin.ignore(std::numeric_limits<std::streamsize>::max(),'\n');
    cout<<"\n\tInvalid year";
    cout<<"\n\tEnter Year : ";
    std::cin >> c;
}

cout<<"\n\tEnter Month : ";
        cin>>b;
        while(std::cin.fail() || b<1 || b>12) {
    std::cin.clear();
    std::cin.ignore(std::numeric_limits<std::streamsize>::max(),'\n');
    cout<<"\n\tInvalid month";
    cout<<"\n\tEnter Month : ";
    std::cin >> b;
}

cout<<"\n\tEnter Date : ";
        cin>>a;
        while(std::cin.fail() || (!dateValidator(a,b,c))) {
    std::cin.clear();
    std::cin.ignore(std::numeric_limits<std::streamsize>::max(),'\n');
    cout<<"\n\tInvalid date";
    cout<<"\n\tEnter Date : ";
    std::cin >> a;
}

car.setCarRentDate(a,b,c,id,1);
cout<<"\n\tYou have rented "<<car.model;
                printCar(car);
                return;

    }else{
        throw "\n\tCar not rented";
    }
                
            }
        }
    }
    
       throw "\n\tNo such car exists!";
}

void Employee::rentaCar(){
    int carId;
    cout << "\n\tEnter car id want to rent : ";
    cin >> carId;
    while(std::cin.fail() || carId<1) {
    std::cin.clear();
    std::cin.ignore(std::numeric_limits<std::streamsize>::max(),'\n');
        std::cout << "\n\tInvaild ID.  Enter a (+) DIGITS : ";
    std::cin >> carId;;
}
if(!checkCarId(carId)){std::cout<<"\n\tNo such car ID exists";
return ;
}
    Car car;
    ifstream read;
    read.open("car.txt");
    while (!read.eof()) {

         read >> car.id;
        read.ignore();
        getline(read, car.model);
        read >> car.dueDateDay;
        read >> car.dueDateMonth;
        read >> car.dueDateYear;
        read >> car.currentCustomerId;
        read >> car.condition;
        read >> car.rent;

        if (car.id == carId) {
            read.close();
            if(customer_record<car.condition){
                cout<<"\n\tLow customer record"<<"\n\t"<<"Need to increase customer record by : "<<car.condition-customer_record<<"\n\tYou cannot rent this car!";
                return;
            }
            else if(car.currentCustomerId>-1){
                cout<<"\n\tAlready rented";
                return;
            }
            else{
                cout<<"\n\tYou can rent "<<car.model;
                int employeeRent= ((85*(car.rent))/100);
            cout<<"\n\tCar rent is : "<<employeeRent;
            cout<<"\n\tNOTE : Fine of Rs.5 per day if car not returned within 30 days";
            cout<<"\n\tRent car"<<car.model<< "\t(y/n)";
            cout<<"\n\tEnter choice : ";
            char choice;
    cin >> choice;
    while(std::cin.fail() || (choice!='Y' && choice!='y'&& choice!='n' && choice!='N')) {
    std::cin.clear();
    std::cin.ignore(std::numeric_limits<std::streamsize>::max(),'\n');
    std::cout << "\n\tInvaild input, Please enter (y/n) : ";
        cout<<"\n\t";
    std::cin >> choice;
    }
//      if(cin.fail()){
//             cin.ignore();
//             throw "\n\tInvalid input";
// }
    if (choice == 'y' || choice =='Y') {
        cout<<"\n\tEnter current date\n\tRange (01/01/1990) to (31/12/9999)";
        int a,b,c;
        cout<<"\n\tEnter Year : ";
        cin>>c;
        while(std::cin.fail() || c<1990 || c>9999) {
    std::cin.clear();
    std::cin.ignore(std::numeric_limits<std::streamsize>::max(),'\n');
    cout<<"\n\tInvalid year";
    cout<<"\n\tEnter Year : ";
    std::cin >> c;
}

cout<<"\n\tEnter Month : ";
        cin>>b;
        while(std::cin.fail() || b<1 || b>12) {
    std::cin.clear();
    std::cin.ignore(std::numeric_limits<std::streamsize>::max(),'\n');
    cout<<"\n\tInvalid month";
    cout<<"\n\tEnter Month : ";
    std::cin >> b;
}

cout<<"\n\tEnter Date : ";
        cin>>a;
        while(std::cin.fail() || (!dateValidator(a,b,c))) {
    std::cin.clear();
    std::cin.ignore(std::numeric_limits<std::streamsize>::max(),'\n');
    cout<<"\n\tInvalid date";
    cout<<"\n\tEnter Date : ";
    std::cin >> a;
}

car.setCarRentDate(a,b,c,id,1);
cout<<"\n\tYou have rented "<<car.model;
                printCar(car);
                return;

    }else{
        throw "\n\tCar not rented";
    }
                
            }
        }
    }
    
       throw "\n\tNo such car exists!";
}



int dayDifference( int d1,int m1, int y1, int d2, int m2,int y2){

int x1,x2;

m1 = (m1 + 9) % 12;
y1 = y1 - m1 / 10;
x1= 365*y1 + y1/4 - y1/100 + y1/400 + (m1*306 + 5)/10 + ( d1 - 1 );

m2 = (m2 + 9) % 12;
y2 = y2 - m2 / 10;
x2= 365*y2 + y2/4 - y2/100 + y2/400 + (m2*306 + 5)/10 + ( d2 - 1 );

if((x2-x1)>=0){
    return x2 - x1;
}
else{
    return -1;
}

}

void Customer::returnRequest(){
    cout<<"Enter ID of car to return : ";
    int returnid;
    cin>>returnid;
     while(std::cin.fail() || returnid<1 || !checkCarId(returnid)) {
    std::cin.clear();
    std::cin.ignore(std::numeric_limits<std::streamsize>::max(),'\n');
    cout<<"\n\tID does not exists";
    cout<<"\n\tEnter valid car ID : ";
    std::cin >> returnid;
}
if(checkCarId2(returnid,id))
{
        Car car;
    ifstream read;
    read.open("car.txt");
    while (!read.eof()) {

         read >> car.id;
        read.ignore();
        getline(read, car.model);
        read >> car.dueDateDay;
        read >> car.dueDateMonth;
        read >> car.dueDateYear;
        read >> car.currentCustomerId;
        read >> car.condition;
        read >> car.rent;

        if (car.id == returnid) {
            //cout<<car.id<<" "<<returnid<<" "<<car.currentCustomerId;
            read.close();
            if(car.currentCustomerId==-1){
                cout<<"\n\tNotrented";
                return;
            }
            else if(car.currentCustomerId==id){
            cout<<"\n\tReturn car "<<car.model<< "\t(y/n)";
            cout<<"\n\tEnter choice : ";
            char choice;
    cin >> choice;
    while(std::cin.fail() || (choice!='Y' && choice!='y'&& choice!='n' && choice!='N')) {
    std::cin.clear();
    std::cin.ignore(std::numeric_limits<std::streamsize>::max(),'\n');
    std::cout << "\n\tInvaild input, Please enter (y/n) : ";
        cout<<"\n\t";
    std::cin >> choice;
    }
//      if(cin.fail()){
//             cin.ignore();
//             throw "\n\tInvalid input";
// }
    if (choice == 'y' || choice =='Y') {
        cout<<"\n\tEnter current date\n\tRange (01/01/1990) to (31/12/9999)";
        int a,b,c;
        cout<<"\n\tEnter Year : ";
        cin>>c;
        while(std::cin.fail() || c<1990 || c>9999) {
    std::cin.clear();
    std::cin.ignore(std::numeric_limits<std::streamsize>::max(),'\n');
    cout<<"\n\tInvalid year";
    cout<<"\n\tEnter Year : ";
    std::cin >> c;
}

cout<<"\n\tEnter Month : ";
        cin>>b;
        while(std::cin.fail() || b<1 || b>12) {
    std::cin.clear();
    std::cin.ignore(std::numeric_limits<std::streamsize>::max(),'\n');
    cout<<"\n\tInvalid month";
    cout<<"\n\tEnter Month : ";
    std::cin >> b;
}

cout<<"\n\tEnter Date : ";
        cin>>a;
        while(std::cin.fail() || (!dateValidator(a,b,c))) {
    std::cin.clear();
    std::cin.ignore(std::numeric_limits<std::streamsize>::max(),'\n');
    cout<<"\n\tInvalid date";
    cout<<"\n\tEnter Date : ";
    std::cin >> a;
}
if(dayDifference(car.dueDateDay,car.dueDateMonth,car.dueDateYear,a,b,c)+1){
    cout<<"\n\tIs car more damaged (y/n)";
            cout<<"\n\tEnter choice : ";
            char choice;
    cin >> choice;
    while(std::cin.fail() || (choice!='Y' && choice!='y'&& choice!='n' && choice!='N')) {
    std::cin.clear();
    std::cin.ignore(std::numeric_limits<std::streamsize>::max(),'\n');
    std::cout << "\n\tInvaild input, Please enter (y/n) : ";
        cout<<"\n\t";
    std::cin >> choice;
    }
    if(choice=='y' || choice=='Y'){
        this->updateCustomerRecord();
    }
    car.setCarRentDate(a,b,c,id,0);
    if(dayDifference(car.dueDateDay,car.dueDateMonth,car.dueDateYear,a,b,c)>30){
         cout<<"\n\tFine : "<<(5*(dayDifference(car.dueDateDay,car.dueDateMonth,car.dueDateYear,a,b,c)-30));
    this->updateFine(5*(dayDifference(car.dueDateDay,car.dueDateMonth,car.dueDateYear,a,b,c)-30));
    this->updateCustomerRecord();
    }
    return;
}
else{
    cout<<"\n\tInvalid date not possible";
    return;
}

    }else{
        throw "\n\tCar not returned";
    }
                
            }
        }
    }
    
       throw "\n\tNo such car exists!";
}

else{
    throw"\n\tYou have not rented this car";
}

}

void Employee::returnRequest(){
    cout<<"Enter ID of car to return : ";
    int returnid;
    cin>>returnid;
     while(std::cin.fail() || returnid<1 || !checkCarId(returnid)) {
    std::cin.clear();
    std::cin.ignore(std::numeric_limits<std::streamsize>::max(),'\n');
    cout<<"\n\tID does not exists";
    cout<<"\n\tEnter valid car ID : ";
    std::cin >> returnid;
}
if(checkCarId2(returnid,id))
{
        Car car;
    ifstream read;
    read.open("car.txt");
    while (!read.eof()) {

         read >> car.id;
        read.ignore();
        getline(read, car.model);
        read >> car.dueDateDay;
        read >> car.dueDateMonth;
        read >> car.dueDateYear;
        read >> car.currentCustomerId;
        read >> car.condition;
        read >> car.rent;

        if (car.id == returnid) {
            //cout<<car.id<<" "<<returnid<<" "<<car.currentCustomerId;
            read.close();
            if(car.currentCustomerId==-1){
                cout<<"\n\tNotrented";
                return;
            }
            else if(car.currentCustomerId==id){
            cout<<"\n\tReturn car "<<car.model<< "\t(y/n)";
            cout<<"\n\tEnter choice : ";
            char choice;
    cin >> choice;
    while(std::cin.fail() || (choice!='Y' && choice!='y'&& choice!='n' && choice!='N')) {
    std::cin.clear();
    std::cin.ignore(std::numeric_limits<std::streamsize>::max(),'\n');
    std::cout << "\n\tInvaild input, Please enter (y/n) : ";
        cout<<"\n\t";
    std::cin >> choice;
    }
//      if(cin.fail()){
//             cin.ignore();
//             throw "\n\tInvalid input";
// }
    if (choice == 'y' || choice =='Y') {
        cout<<"\n\tEnter current date\n\tRange (01/01/1990) to (31/12/9999)";
        int a,b,c;
        cout<<"\n\tEnter Year : ";
        cin>>c;
        while(std::cin.fail() || c<1990 || c>9999) {
    std::cin.clear();
    std::cin.ignore(std::numeric_limits<std::streamsize>::max(),'\n');
    cout<<"\n\tInvalid year";
    cout<<"\n\tEnter Year : ";
    std::cin >> c;
}

cout<<"\n\tEnter Month : ";
        cin>>b;
        while(std::cin.fail() || b<1 || b>12) {
    std::cin.clear();
    std::cin.ignore(std::numeric_limits<std::streamsize>::max(),'\n');
    cout<<"\n\tInvalid month";
    cout<<"\n\tEnter Month : ";
    std::cin >> b;
}

cout<<"\n\tEnter Date : ";
        cin>>a;
        while(std::cin.fail() || (!dateValidator(a,b,c))) {
    std::cin.clear();
    std::cin.ignore(std::numeric_limits<std::streamsize>::max(),'\n');
    cout<<"\n\tInvalid date";
    cout<<"\n\tEnter Date : ";
    std::cin >> a;
}
if(dayDifference(car.dueDateDay,car.dueDateMonth,car.dueDateYear,a,b,c)+1){
    cout<<"\n\tIs car more damaged (y/n)";
            cout<<"\n\tEnter choice : ";
            char choice;
    cin >> choice;
    while(std::cin.fail() || (choice!='Y' && choice!='y'&& choice!='n' && choice!='N')) {
    std::cin.clear();
    std::cin.ignore(std::numeric_limits<std::streamsize>::max(),'\n');
    std::cout << "\n\tInvaild input, Please enter (y/n) : ";
        cout<<"\n\t";
    std::cin >> choice;
    }
    if(choice=='y' || choice=='Y'){
        this->updateEmployeeRecord();
    }
    car.setCarRentDate(a,b,c,id,0);
    if(dayDifference(car.dueDateDay,car.dueDateMonth,car.dueDateYear,a,b,c)>30){
        cout<<"\n\tFine : "<<(5*(dayDifference(car.dueDateDay,car.dueDateMonth,car.dueDateYear,a,b,c)-30));
    this->updateFine(5*(dayDifference(car.dueDateDay,car.dueDateMonth,car.dueDateYear,a,b,c)-30));
    this->updateEmployeeRecord();
    }
    return;
}
else{
    cout<<"\n\tInvalid date not possible";
    return;
}

    }else{
        throw "\n\tCar not returned";
    }
                
            }
        }
    }
    
       throw "\n\tNo such car exists!";
}

else{
    throw"\n\tYou have not rented this car";
}

}



void Customer::rentedCarData(){
    int numberofCars=0;
    Car car;
    ifstream read;
    read.open("car.txt");
    while (!read.eof()) {

         read >> car.id;
        read.ignore();
        getline(read, car.model);
        read >> car.dueDateDay;
        read >> car.dueDateMonth;
        read >> car.dueDateYear;
        read >> car.currentCustomerId;
        read >> car.condition;
        read >> car.rent;

        // read >> student.id;
        // read.ignore();
        // getline(read, student.name);
        // read >> student.age;
        // read >> student.degree;
        // read >> student.semesterNo;
        if (car.currentCustomerId == id) {
            numberofCars++;
            printCar(car);
            // return id;
        }
    }
    cout<<"\n\tTotal number of rented cars by "<<name<<" : "<<numberofCars<<endl;
    //    throw "No such car exists!";
}

void Employee::rentedCarData(){
    int numberofCars=0;
    Car car;
    ifstream read;
    read.open("car.txt");
    while (!read.eof()) {

         read >> car.id;
        read.ignore();
        getline(read, car.model);
        read >> car.dueDateDay;
        read >> car.dueDateMonth;
        read >> car.dueDateYear;
        read >> car.currentCustomerId;
        read >> car.condition;
        read >> car.rent;
        int employeeRent= (((85*car.rent)/100));
        car.rent=employeeRent;

        // read >> student.id;
        // read.ignore();
        // getline(read, student.name);
        // read >> student.age;
        // read >> student.degree;
        // read >> student.semesterNo;
        if (car.currentCustomerId == id) {
            numberofCars++;
            printCar(car);
            // return id;
        }
    }
    cout<<"\n\tTotal number of rented cars by "<<name<<" : "<<numberofCars<<endl;
    //    throw "No such car exists!";
}

void sumOfRecordsCustomer(int a[2]){
     Customer customer;
    ifstream read;
    a[0]=0;
    a[1]=0;
    //a[0] number of customers
    //a[1] sum of records
    read.open("customer.txt");
    while (!read.eof()) {
        read >> customer.id;
        read >> customer.password;
        read.ignore();
        getline(read, customer.rentedCars);
        getline(read, customer.name);
        read >> customer.fineDue;
        // read >> customer.degree;
        read >> customer.customer_record;
       a[1]= a[1]+customer.customer_record;
       a[0]++;
    }
    
    // cout<<a[0]<<" "<<a[1]<<endl;
    read.close();
    return;   
}

void sumOfRecordsEmployee(int a[2]){
     Employee customer;
    ifstream read;
    a[0]=0;
    a[1]=0;
    //a[0] number of customers
    //a[1] sum of records
    read.open("employee.txt");
    while (!read.eof()) {
        read >> customer.id;
        read >> customer.password;
        read.ignore();
        getline(read, customer.rentedCars);
        getline(read, customer.name);
        read >> customer.fineDue;
        // read >> customer.degree;
        read >> customer.customer_record;
       a[1]= a[1]+customer.customer_record;
       a[0]++;
    }
    
    // cout<<a[0]<<" "<<a[1]<<endl;
    read.close();
    return;   
}



void addCustomer() {
    Customer customer;
    int userInput;
    int sumOfRecord[2];
   
         cout << "\n\tEnter customer ID : ";
         cin>>customer.id;;
         
while(std::cin.fail() || customer.id<1 || checkUser(customer.id)) {
    std::cin.clear();
    std::cin.ignore(std::numeric_limits<std::streamsize>::max(),'\n');
    if(checkUser(customer.id)){std::cout<<"\n\tUser ID already exists \n\tEnter new ID: ";}
    else{
        std::cout << "\n\tInvaild entry.  Enter a (+) DIGITS : ";
    }
    std::cin >> customer.id;;
}
std::cout << "\n\tID = " << customer.id << std::endl;   
    // cout << "\n\tEnter customer ID : ";
    // cin >> customer.id;
    cout << "\n\tEnter customer Password : ";
         cin>>customer.password;;
         
while(std::cin.fail() || customer.password<1 ) {
    std::cin.clear();
    std::cin.ignore(std::numeric_limits<std::streamsize>::max(),'\n');
    std::cout << "\n\tInvaild entry.  Enter a (+) DIGITS : ";
    std::cin >> customer.password;;
}
    sumOfRecordsCustomer(sumOfRecord);
    int avg;
    if(sumOfRecord[1]/sumOfRecord[0]>1000000000){
        cout<<"\n\tNo customer data, default customer record: 100";
        avg =100;}
    else{
        avg = sumOfRecord[1]/sumOfRecord[0];
        
    }
    cout << "\n\tPress Enter to continue : ";
    cin.get();
    getline(cin, customer.rentedCars); //Nouman Habib
    if(cin.fail()){
            cin.ignore();
            throw "\n\tInvalid input";
}

     cout << "\n\tEnter customer Name : ";
    
    getline(cin, customer.name); //Nouman Habib
    while(std::cin.fail() ) {
    std::cin.clear();
    std::cin.ignore(std::numeric_limits<std::streamsize>::max(),'\n');
    std::cout << "\n\tInvaild entry.  Enter customer name : ";
    std::cin >> customer.name;
}

    cout << "\n\tEnter customer fineDue : ";
    cin >> customer.fineDue;
    while(std::cin.fail() || customer.fineDue<0) {
    std::cin.clear();
    std::cin.ignore(std::numeric_limits<std::streamsize>::max(),'\n');
    std::cout << "\n\tInvaild entry, Please enter (+)Number";
        cout<<"\n\t";
    std::cin >> customer.fineDue;
    }
    // cout << "\n\tEnter customer degree : ";
    // cin >> customer.degree;
    // cout << "\n\tEnter customer record : ";
    customer.customer_record = avg;
    // ID++;

    ofstream write;
    write.open("customer.txt", ios::app);
    write << "\n" << customer.id;
    write << "\n" << customer.password;
    write << "\n" << customer.rentedCars ;
    write<< "\n" << customer.name;
    write << "\n" << customer.fineDue ;

    // write << "\n" << customer.degree ;
    write << "\n" << customer.customer_record;
    write.close();
    // write.open("id.txt");
    // write << ID;
    // write.close();
    
    cout << "\n\tData save to file";
}

void addEmployee() {
    Employee customer;
    int userInput;
    int sumOfRecord[2];
   
         cout << "\n\tEnter employee ID : ";
         cin>>customer.id;;
         
while(std::cin.fail() || customer.id<1 || checkUser(customer.id)) {
    std::cin.clear();
    std::cin.ignore(std::numeric_limits<std::streamsize>::max(),'\n');
    if(checkUser(customer.id)){std::cout<<"\n\tUser ID already exists \n\tEnter new ID: ";}
    else{
        std::cout << "\n\tInvaild entry.  Enter a (+) DIGITS : ";
    }
    std::cin >> customer.id;;
}
std::cout << "\n\tID = " << customer.id << std::endl;   
    // cout << "\n\tEnter customer ID : ";
    // cin >> customer.id;
    cout << "\n\tEnter employee Password : ";
         cin>>customer.password;;
         
while(std::cin.fail() || customer.password<1 ) {
    std::cin.clear();
    std::cin.ignore(std::numeric_limits<std::streamsize>::max(),'\n');
    std::cout << "\n\tInvaild entry.  Enter a (+) DIGITS : ";
    std::cin >> customer.password;;
}
    sumOfRecordsEmployee(sumOfRecord);
    int avg;
    if(sumOfRecord[1]/sumOfRecord[0]>1000000000){
        cout<<"\n\tNo employee data, default employee record: 100";
        avg =100;}
    else{
        avg = sumOfRecord[1]/sumOfRecord[0];
        
    }
    cout << "\n\tPress Enter to continue : ";
    cin.get();
    getline(cin, customer.rentedCars); //Nouman Habib
    if(cin.fail()){
            cin.ignore();
            throw "\n\tInvalid input";
}

     cout << "\n\tEnter employee Name : ";
    
    getline(cin, customer.name); //Nouman Habib
    while(std::cin.fail() ) {
    std::cin.clear();
    std::cin.ignore(std::numeric_limits<std::streamsize>::max(),'\n');
    std::cout << "\n\tInvaild entry.  Enter employee name : ";
    std::cin >> customer.name;
}

    cout << "\n\tEnter employee fineDue : ";
    cin >> customer.fineDue;
    while(std::cin.fail() || customer.fineDue<0) {
    std::cin.clear();
    std::cin.ignore(std::numeric_limits<std::streamsize>::max(),'\n');
    std::cout << "\n\tInvaild entry, Please enter (+)Number";
        cout<<"\n\t";
    std::cin >> customer.fineDue;
    }
    // cout << "\n\tEnter customer degree : ";
    // cin >> customer.degree;
    // cout << "\n\tEnter customer record : ";
    customer.customer_record = avg;
    // ID++;

    ofstream write;
    write.open("employee.txt", ios::app);
    write << "\n" << customer.id;
    write << "\n" << customer.password;
    write << "\n" << customer.rentedCars ;
    write<< "\n" << customer.name;
    write << "\n" << customer.fineDue ;

    // write << "\n" << customer.degree ;
    write << "\n" << customer.customer_record;
    write.close();
    // write.open("id.txt");
    // write << ID;
    // write.close();
    
    cout << "\n\tData save to file";
}





void readDataCustomer() { 
    
    // if(dataExistChecker()==0){
    //     throw "\n\tAdd customers";
    // }
    
    int customerNumber=0;
    Customer customer;
    ifstream read;
    read.open("customer.txt");
    while (!read.eof()) {
        read >> customer.id;
        read>> customer.password;
        read.ignore();
        getline(read, customer.rentedCars);
        getline(read, customer.name);

        read >> customer.fineDue;
        // read >> customer.degree;
        read >> customer.customer_record;
        customer.printCustomer();
        customerNumber++;

    }
    read.close();    
    cout<<"\n\tTotal number of customer(s) : "<<customerNumber<<endl;     
}

void readDataEmployee() { 
    
    // if(dataExistChecker()==0){
    //     throw "\n\tAdd customers";
    // }
    
    int customerNumber=0;
    Employee customer;
    ifstream read;
    read.open("employee.txt");
    while (!read.eof()) {
        read >> customer.id;
        read>> customer.password;
        read.ignore();
        getline(read, customer.rentedCars);
        getline(read, customer.name);

        read >> customer.fineDue;
        // read >> customer.degree;
        read >> customer.customer_record;
        customer.printEmployee();
        customerNumber++;

    }
    read.close();    
    cout<<"\n\tTotal number of employee(s) : "<<customerNumber<<endl;     
}


void Customer::setCustomer(int i){
    id = i;

    std::cout << "\n\tID = " <<id << std::endl; 
    Customer customer2;
    ifstream read;
    read.open("customer.txt");
    while (!read.eof()) {
        read >> customer2.id;
        read>> customer2.password;
        read.ignore();
        getline(read, customer2.rentedCars)
;
        getline(read, customer2.name);

        read >> customer2.fineDue;
        // read >> customer.degree;
        read >> customer2.customer_record;
        if (customer2.id == id) {
            id = customer2.id;
            password= customer2.password;
            rentedCars= customer2.rentedCars;
            name=customer2.name;
            fineDue=customer2.fineDue;
            customer_record=customer2.customer_record;
          
        }
    }
    read.close();

}


void Employee::setEmployee(int i){
    id = i;

    std::cout << "\n\tID = " <<id << std::endl; 
    Employee customer2;
    ifstream read;
    read.open("employee.txt");
    while (!read.eof()) {
        read >> customer2.id;
        read>> customer2.password;
        read.ignore();
        getline(read, customer2.rentedCars)
;
        getline(read, customer2.name);

        read >> customer2.fineDue;
        // read >> customer.degree;
        read >> customer2.customer_record;
        if (customer2.id == id) {
            id = customer2.id;
            password= customer2.password;
            rentedCars= customer2.rentedCars;
            name=customer2.name;
            fineDue=customer2.fineDue;
            customer_record=customer2.customer_record;
          
        }
    }
    read.close();

}


int searchDataCustomer() {
    
    // if(dataExistChecker()==0){
    //     throw "\n\tAdd customers";
    // }
    int id;
    cout << "\n\tEnter customer id : ";
    cin >> id;
    while(std::cin.fail() || id<1 ) {
    std::cin.clear();
    std::cin.ignore(std::numeric_limits<std::streamsize>::max(),'\n');
    if(std::cin.fail() ){std::cout << "\n\tInvaild entry,  Enter a (+) DIGITS  : ";
    std::cin >> id;
    }
    else if( id<1){std::cout << "\n\tInvaild entry,  Enter a (+) DIGITS  : ";
    std::cin >> id;
    }

    else if(!checkUser(id)){
        throw "\n\tNo such user exists!";
        
    }
   
    
}


std::cout << "\n\tID = " <<id << std::endl; 
    Customer customer;
    ifstream read;
    read.open("customer.txt");
    while (!read.eof()) {
        read >> customer.id;
        read>> customer.password;
        read.ignore();
        getline(read, customer.rentedCars)
;
        getline(read, customer.name);

        read >> customer.fineDue;
        // read >> customer.degree;
        read >> customer.customer_record;
        if (customer.id == id) {
            customer.printCustomer();
            return id;
        }
    }
    read.close();
    throw "\n\tNo such customer found";
    
}


int searchDataEmployee() {
    
    // if(dataExistChecker()==0){
    //     throw "\n\tAdd customers";
    // }
    int id;
    cout << "\n\tEnter employee id : ";
    cin >> id;
    while(std::cin.fail() || id<1 ) {
    std::cin.clear();
    std::cin.ignore(std::numeric_limits<std::streamsize>::max(),'\n');
    if(std::cin.fail() ){std::cout << "\n\tInvaild entry,  Enter a (+) DIGITS  : ";
    std::cin >> id;
    }
    else if( id<1){std::cout << "\n\tInvaild entry,  Enter a (+) DIGITS  : ";
    std::cin >> id;
    }

    else if(!checkUser(id)){
        throw "\n\tNo such user exists!";
        
    }
   
    
}


std::cout << "\n\tID = " <<id << std::endl; 
    Employee customer;
    ifstream read;
    read.open("employee.txt");
    while (!read.eof()) {
        read >> customer.id;
        read>> customer.password;
        read.ignore();
        getline(read, customer.rentedCars)
;
        getline(read, customer.name);

        read >> customer.fineDue;
        // read >> customer.degree;
        read >> customer.customer_record;
        if (customer.id == id) {
            customer.printEmployee();
            return id;
        }
    }
    read.close();
    throw "\n\tNo such employee found";
    
}


void updatedCarUser(int id){
    Car car;
        ofstream tempFile;
        tempFile.open("temp.txt");
        ifstream read;
        read.open("car.txt");
        while (!read.eof()) {
            read >> car.id;
            read.ignore();
            getline(read, car.model);
            read >> car.dueDateDay;
            read >> car.dueDateMonth;
            read >> car.dueDateYear;
            // read >> customer.degree;
            read >> car.currentCustomerId;
            read >> car.condition;
            read >> car.rent;
            if (car.currentCustomerId != id) {
                tempFile << "\n" << car.id;
                tempFile << "\n" << car.model;
                tempFile << "\n" << car.dueDateDay;
                tempFile << "\n" << car.dueDateMonth;
                tempFile << "\n" << car.dueDateYear;
                tempFile << "\n" << car.currentCustomerId;
                tempFile << "\n" << car.condition;
                tempFile << "\n" << car.rent;
            }
            else {
                    tempFile << "\n"<< car.id;
                tempFile << "\n" << car.model;
                     tempFile << "\n" << -1;
                tempFile << "\n" << -1;
                tempFile << "\n" << -1;
                tempFile << "\n" << -1;
                tempFile << "\n" << car.condition;
                tempFile << "\n" << car.rent;
            }
        }
        read.close();
        tempFile.close();
        remove("car.txt");
        rename("temp.txt", "car.txt");
        cout << "\n\tData updated successfuly";
     
}

void deleteCustomerData() {
    int id;
    
    try{
        id = searchDataCustomer();
        
    } catch(const char* msg){
        throw "\n\tNo such customer found";
        cerr<<msg<<endl;
        
    }
    cout << "\n\tDelete record (y/n) : ";
    char choice;
    cin >> choice;
    while(std::cin.fail() || (choice!='Y' && choice!='y'&& choice!='n' && choice!='N')) {
    std::cin.clear();
    std::cin.ignore(std::numeric_limits<std::streamsize>::max(),'\n');
    std::cout << "\n\tInvaild input, Please enter (y/n) : ";
        cout<<"\n\t";
    std::cin >> choice;
    }
    if (choice == 'y') {
        updatedCarUser(id);
        Customer customer;
        ofstream tempFile;
        tempFile.open("temp.txt");
        ifstream read;
        read.open("customer.txt");
        while (!read.eof()) {
            read >> customer.id;
            read>> customer.password;
            read.ignore();
            getline(read, customer.rentedCars);
            getline(read, customer.name);

            read >> customer.fineDue;
            // read >> customer.degree;
            read >> customer.customer_record;
            if (customer.id != id) {
                tempFile << "\n" << customer.id;
                tempFile << "\n" << customer.password;
                tempFile << "\n" << customer.rentedCars;
                tempFile << "\n" << customer.name;
        
                tempFile << "\n" << customer.fineDue;
                // tempFile << "\n" << customer.degree;
                tempFile << "\n" << customer.customer_record;
            }
        }
        read.close();
        tempFile.close();
        remove("customer.txt");
        rename("temp.txt", "customer.txt");
        cout << "\n\tData deleted successfuly";
    }
    else {
        cout << "\n\tRecord not deleted";
    }
}


void deleteEmployeeData() {
    int id;
    
    try{
        id = searchDataEmployee();
        
    } catch(const char* msg){
        throw "\n\tNo such employee found";
        cerr<<msg<<endl;
        
    }
    cout << "\n\tDelete record (y/n) : ";
    char choice;
    cin >> choice;
    while(std::cin.fail() || (choice!='Y' && choice!='y'&& choice!='n' && choice!='N')) {
    std::cin.clear();
    std::cin.ignore(std::numeric_limits<std::streamsize>::max(),'\n');
    std::cout << "\n\tInvaild input, Please enter (y/n) : ";
        cout<<"\n\t";
    std::cin >> choice;
    }
    if (choice == 'y') {
        updatedCarUser(id);
        Employee customer;
        ofstream tempFile;
        tempFile.open("temp.txt");
        ifstream read;
        read.open("employee.txt");
        while (!read.eof()) {
            read >> customer.id;
            read>> customer.password;
            read.ignore();
            getline(read, customer.rentedCars);
            getline(read, customer.name);

            read >> customer.fineDue;
            // read >> customer.degree;
            read >> customer.customer_record;
            if (customer.id != id) {
                tempFile << "\n" << customer.id;
                tempFile << "\n" << customer.password;
                tempFile << "\n" << customer.rentedCars;
                tempFile << "\n" << customer.name;
        
                tempFile << "\n" << customer.fineDue;
                // tempFile << "\n" << customer.degree;
                tempFile << "\n" << customer.customer_record;
            }
        }
        read.close();
        tempFile.close();
        remove("employee.txt");
        rename("temp.txt", "employee.txt");
        cout << "\n\tData deleted successfuly";
    }
    else {
        cout << "\n\tRecord not deleted";
    }
}

void updateDataCustomer() {
    int id;
    try{
        id = searchDataCustomer();
    } catch(const char* msg){
                cerr<<msg<<endl;
                return;
    }
    
    if(checkUser(id)){cout << "\n\tYou want to update record (y/n) : ";
    char choice;
    int option;
    cin >> choice;
    while(std::cin.fail() || (choice!='Y' && choice!='y'&& choice!='n' && choice!='N')) {
    std::cin.clear();
    std::cin.ignore(std::numeric_limits<std::streamsize>::max(),'\n');
    std::cout << "\n\tInvaild input, Please enter (y/n) : ";
        cout<<"\n\t";
    std::cin >> choice;
    }
//      if(cin.fail()){
//             cin.ignore();
//             throw "\n\tInvalid input";
// }
    if (choice == 'y' || choice =='Y') {
        Customer newData;
        cout<<"\n\tWhat do you will like to update : ";
        // cout<<"\n\t1. Rented Cars";
        cout<<"\n\t1. Fine Due";
        cout<<"\n\t2. Customer record";
        cout<<"\n\t3. Customer ID";
        cout<<"\n\t4. Customer Password";
        cout<<"\n\t5. Customer Name";
        cout<<"\n\t";
        cin>>option;
        while(std::cin.fail() || option<1 || option>5) {
    std::cin.clear();
    std::cin.ignore(std::numeric_limits<std::streamsize>::max(),'\n');
    std::cout << "\n\tInvaild entry,";
     cout<<"\n\tWhat do you will like to update : ";
        // cout<<"\n\t1. Rented Cars";
        cout<<"\n\t1. Fine Due";
        cout<<"\n\t2. Customer record";
        cout<<"\n\t3. Customer ID";
        cout<<"\n\t4. Customer Password";
        cout<<"\n\t5. Customer Name";
        cout<<"\n\t";
    std::cin >> option;
    }
        
        if(option==1){
            cout << "\n\tEnter new customer fineDue : ";
        cin >> newData.fineDue;
         while(std::cin.fail() || newData.fineDue<0) {
    std::cin.clear();
    std::cin.ignore(std::numeric_limits<std::streamsize>::max(),'\n');
    std::cout << "\n\tInvaild entry, Please enter (+)Number";
        cout<<"\n\t";
    std::cin >> newData.fineDue;
    }
        }
        
        // cout << "\n\tEnter customer degree : ";
        // cin >> newData.degree;
        else if(option==2){
            cout << "\n\tEnter new Customer Record (0 - 100) : ";
         cin >> newData.customer_record;
         while(std::cin.fail() || newData.customer_record<0 || newData.customer_record>100) {
    std::cin.clear();
    std::cin.ignore(std::numeric_limits<std::streamsize>::max(),'\n');
    std::cout << "\n\tInvaild entry, Please enter (+)Number (0 - 100) : ";
        cout<<"\n\t";
    std::cin >> newData.customer_record;
    }
        }
        else if(option==3){
            cout << "\n\tEnter new customer ID : ";
        cin >> newData.id;
        while(std::cin.fail() || newData.id<1 || checkUser(newData.id)) {
    std::cin.clear();
    std::cin.ignore(std::numeric_limits<std::streamsize>::max(),'\n');
    if(checkUser(newData.id)){std::cout<<"\n\tUser ID already exists \n\tEnter new ID: ";}
    else{
        std::cout << "\n\tInvaild entry.  Enter a (+) DIGITS : ";
    }
    std::cin >> newData.id;;
}
        }
        else if(option==4){
            cout << "\n\tEnter new customer Password : ";
        cin >> newData.password;
        while(std::cin.fail() || newData.password<1 ) {
    std::cin.clear();
    std::cin.ignore(std::numeric_limits<std::streamsize>::max(),'\n');
    
        std::cout << "\n\tInvaild entry.  Enter a (+) DIGITS : ";
    std::cin >> newData.password;
}
        }
        else if(option==5){
            cout << "\n\tEnter new customer name : ";
        cin.get();

        getline(cin, newData.name);
        if(cin.fail()){
            cin.ignore();
            throw "\n\tInvalid input";
}
        }
         else{
        //     cout<<"\tInvalid input\n";
        throw "\n\tInvalid Input";
        //     return;
         }
        Customer customer;
        ofstream tempFile;
        tempFile.open("temp.txt");
        ifstream read;
        read.open("customer.txt");
        while (!read.eof()) {
            read >> customer.id;
                        read >> customer.password;

            read.ignore();
            getline(read, customer.rentedCars);
            getline(read, customer.name);

            read >> customer.fineDue;
            // read >> customer.degree;
            read >> customer.customer_record;
            if (customer.id != id) {
                tempFile << "\n" << customer.id;
                tempFile << "\n" << customer.password;
                tempFile << "\n" << customer.rentedCars;        
                tempFile << "\n" << customer.name;        
                tempFile << "\n" << customer.fineDue;
                // tempFile << "\n" << customer.degree;
                tempFile << "\n" << customer.customer_record;
            }
            else {
                   
                if(option==1){
                    tempFile << "\n"<< customer.id;
                    tempFile << "\n" << customer.password;
                    tempFile << "\n" << customer.rentedCars;
                     tempFile << "\n" << customer.name;  
                    tempFile << "\n"<< newData.fineDue;
                    tempFile << "\n" << customer.customer_record;
                }
                // tempFile << "\n" << newData.degree;
                else if(option==2){
                    tempFile << "\n"<< customer.id;
                    tempFile << "\n" << customer.password;
                    tempFile << "\n" << customer.rentedCars;
                     tempFile << "\n" << customer.name;  
                    tempFile << "\n" << customer.fineDue;
                    tempFile << "\n"<< newData.customer_record;
                }
                else if(option==3){
                    tempFile << "\n"<< newData.id;
                    tempFile << "\n" << customer.password;
                    tempFile << "\n" << customer.rentedCars;
                     tempFile << "\n" << customer.name;  
                    tempFile << "\n" << customer.fineDue;
                    tempFile << "\n"<< customer.customer_record;
                }
                else if(option==4){
                    tempFile << "\n"<< customer.id;
                    tempFile << "\n" << newData.password;
                    tempFile << "\n" << customer.rentedCars;
                     tempFile << "\n" << customer.name;  
                    tempFile << "\n" << customer.fineDue;
                    tempFile << "\n"<< customer.customer_record;
                }
                else if(option==5){
                    tempFile << "\n"<< customer.id;
                    tempFile << "\n" << customer.password;
                    tempFile << "\n" << customer.rentedCars;
                     tempFile << "\n" << newData.name;  
                    tempFile << "\n" << customer.fineDue;
                    tempFile << "\n"<< customer.customer_record;
                }
            }
        }
        read.close();
        tempFile.close();
        remove("customer.txt");
        rename("temp.txt", "customer.txt");
        cout << "\n\tData updated successfuly";
    }
    else  if (choice == 'n' || choice =='N') {
        cout<<"\n\tNo change implemented";
        
    }
    else{
        throw "\n\tRecord not updated";
    }
    }
}

void updateDataEmployee() {
    int id;
    try{
        id = searchDataEmployee();
    } catch(const char* msg){
                cerr<<msg<<endl;
                return;
    }
    // cout<<"test1";
    if(checkUser(id)){cout << "\n\tYou want to update record (y/n) : ";
    char choice;
    int option;
    cin >> choice;
    while(std::cin.fail() || (choice!='Y' && choice!='y'&& choice!='n' && choice!='N')) {
    std::cin.clear();
    std::cin.ignore(std::numeric_limits<std::streamsize>::max(),'\n');
    std::cout << "\n\tInvaild input, Please enter (y/n) : ";
        cout<<"\n\t";
    std::cin >> choice;
    }
//      if(cin.fail()){
//             cin.ignore();
//             throw "\n\tInvalid input";
// }
    if (choice == 'y' || choice =='Y') {
        Employee newData;
        cout<<"\n\tWhat do you will like to update : ";
        // cout<<"\n\t1. Rented Cars";
        cout<<"\n\t1. Fine Due";
        cout<<"\n\t2. Employee record";
        cout<<"\n\t3. Employee ID";
        cout<<"\n\t4. Employee Password";
        cout<<"\n\t5. Employee Name";
        cout<<"\n\tEnter choice : ";
        // cout<<"\n\t";
        cin>>option;
        while(std::cin.fail() || option<1 || option>5) {
    std::cin.clear();
    std::cin.ignore(std::numeric_limits<std::streamsize>::max(),'\n');
    std::cout << "\n\tInvaild entry,";
     cout<<"\n\tWhat do you will like to update : ";
        // cout<<"\n\t1. Rented Cars";
        cout<<"\n\t1. Fine Due";
        cout<<"\n\t2. Employee record";
        cout<<"\n\t3. Employee ID";
        cout<<"\n\t4. Employee Password";
        cout<<"\n\t5. Employee Name";
        //cout<<"\n\t";
        cout<<"\n\tEnter choice : ";
    std::cin >> option;
    }
        
        if(option==1){
            cout << "\n\tEnter new employee fineDue : ";
        cin >> newData.fineDue;
         while(std::cin.fail() || newData.fineDue<0) {
    std::cin.clear();
    std::cin.ignore(std::numeric_limits<std::streamsize>::max(),'\n');
    std::cout << "\n\tInvaild entry, Please enter (+)Number";
        cout<<"\n\t";
    std::cin >> newData.fineDue;
    }
        }
        
        // cout << "\n\tEnter customer degree : ";
        // cin >> newData.degree;
        else if(option==2){
            cout << "\n\tEnter new Employee Record (0 - 100) : ";
         cin >> newData.customer_record;
         while(std::cin.fail() || newData.customer_record<0 || newData.customer_record>100) {
    std::cin.clear();
    std::cin.ignore(std::numeric_limits<std::streamsize>::max(),'\n');
    std::cout << "\n\tInvaild entry, Please enter (+)Number (0 - 100) : ";
        cout<<"\n\t";
    std::cin >> newData.customer_record;
    }
        }
        else if(option==3){
            cout << "\n\tEnter new employee ID : ";
        cin >> newData.id;
        while(std::cin.fail() || newData.id<1 || checkUser(newData.id)) {
    std::cin.clear();
    std::cin.ignore(std::numeric_limits<std::streamsize>::max(),'\n');
    if(checkUser(newData.id)){std::cout<<"\n\tUser ID already exists \n\tEnter new ID: ";}
    else{
        std::cout << "\n\tInvaild entry.  Enter a (+) DIGITS : ";
    }
    std::cin >> newData.id;;
}
        }
        else if(option==4){
            cout << "\n\tEnter new employee Password : ";
        cin >> newData.password;
        while(std::cin.fail() || newData.password<1 ) {
    std::cin.clear();
    std::cin.ignore(std::numeric_limits<std::streamsize>::max(),'\n');
    
        std::cout << "\n\tInvaild entry.  Enter a (+) DIGITS : ";
    std::cin >> newData.password;
}
        }
        else if(option==5){
            cout << "\n\tEnter new employee name : ";
        cin.get();

        getline(cin, newData.name);
        if(cin.fail()){
            cin.ignore();
            throw "\n\tInvalid input";
}
        }
         else{
        //     cout<<"\tInvalid input\n";
        throw "\n\tInvalid Input";
        //     return;
         }
        Employee customer;
        ofstream tempFile;
        tempFile.open("temp.txt");
        ifstream read;
        read.open("employee.txt");
        while (!read.eof()) {
            read >> customer.id;
                        read >> customer.password;

            read.ignore();
            getline(read, customer.rentedCars);
            getline(read, customer.name);

            read >> customer.fineDue;
            // read >> customer.degree;
            read >> customer.customer_record;
            if (customer.id != id) {
                tempFile << "\n" << customer.id;
                tempFile << "\n" << customer.password;
                tempFile << "\n" << customer.rentedCars;        
                tempFile << "\n" << customer.name;        
                tempFile << "\n" << customer.fineDue;
                // tempFile << "\n" << customer.degree;
                tempFile << "\n" << customer.customer_record;
            }
            else {
                   
                if(option==1){
                    tempFile << "\n"<< customer.id;
                    tempFile << "\n" << customer.password;
                    tempFile << "\n" << customer.rentedCars;
                     tempFile << "\n" << customer.name;  
                    tempFile << "\n"<< newData.fineDue;
                    tempFile << "\n" << customer.customer_record;
                }
                // tempFile << "\n" << newData.degree;
                else if(option==2){
                    tempFile << "\n"<< customer.id;
                    tempFile << "\n" << customer.password;
                    tempFile << "\n" << customer.rentedCars;
                     tempFile << "\n" << customer.name;  
                    tempFile << "\n" << customer.fineDue;
                    tempFile << "\n"<< newData.customer_record;
                }
                else if(option==3){
                    tempFile << "\n"<< newData.id;
                    tempFile << "\n" << customer.password;
                    tempFile << "\n" << customer.rentedCars;
                     tempFile << "\n" << customer.name;  
                    tempFile << "\n" << customer.fineDue;
                    tempFile << "\n"<< customer.customer_record;
                }
                else if(option==4){
                    tempFile << "\n"<< customer.id;
                    tempFile << "\n" << newData.password;
                    tempFile << "\n" << customer.rentedCars;
                     tempFile << "\n" << customer.name;  
                    tempFile << "\n" << customer.fineDue;
                    tempFile << "\n"<< customer.customer_record;
                }
                else if(option==5){
                    tempFile << "\n"<< customer.id;
                    tempFile << "\n" << customer.password;
                    tempFile << "\n" << customer.rentedCars;
                     tempFile << "\n" << newData.name;  
                    tempFile << "\n" << customer.fineDue;
                    tempFile << "\n"<< customer.customer_record;
                }
            }
        }
        read.close();
        tempFile.close();
        remove("employee.txt");
        rename("temp.txt", "employee.txt");
        cout << "\n\tData updated successfuly";
    }
    else  if (choice == 'n' || choice =='N') {
        cout<<"\n\tNo change implemented";
        
    }
    else{
        throw "\n\tRecord not updated";
    }
    }
}



void clearDueCustomer(){
    
    int id;
    try{
        id = searchDataCustomer();
    } catch(const char* msg){
                cerr<<msg<<endl;
                return;
    }
    
    if(checkUser(id)){cout << "\n\tYou want to clear due (y/n) : ";
    char choice;
    int option = 2;
    cin >> choice;
    while(std::cin.fail() || (choice!='Y' && choice!='y'&& choice!='n' && choice!='N')) {
    std::cin.clear();
    std::cin.ignore(std::numeric_limits<std::streamsize>::max(),'\n');
    std::cout << "\n\tInvaild input, Please enter (y/n) : ";
        cout<<"\n\t";
    std::cin >> choice;
    }

    if (choice == 'y' || choice =='Y') {
        Customer newData;

        
           if(option==2){
          
        newData.fineDue=0;
         while(std::cin.fail() || newData.fineDue<0) {
    std::cin.clear();
    std::cin.ignore(std::numeric_limits<std::streamsize>::max(),'\n');
    std::cout << "\n\tInvaild entry, Please enter (+)Number";
        cout<<"\n\t";
    std::cin >> newData.fineDue;
    }
        }

        Customer customer;
        ofstream tempFile;
        tempFile.open("temp.txt");
        ifstream read;
        read.open("customer.txt");
        while (!read.eof()) {
            read >> customer.id;
            read >> customer.password;

            read.ignore();
            getline(read, customer.rentedCars);
            getline(read, customer.name);

            read >> customer.fineDue;
            // read >> customer.degree;
            read >> customer.customer_record;
            if (customer.id != id) {
                tempFile << "\n" << customer.id;
                tempFile << "\n" << customer.password;
                tempFile << "\n" << customer.rentedCars;
        tempFile << "\n" << customer.name;
                tempFile << "\n" << customer.fineDue;
                // tempFile << "\n" << customer.degree;
                tempFile << "\n" << customer.customer_record;
            }
            else {
                if(option==2){
                    tempFile << "\n"<< customer.id;
                    tempFile << "\n" << customer.password;
                    tempFile << "\n" << customer.rentedCars;
                    tempFile << "\n" << customer.name;
                    tempFile << "\n"<< newData.fineDue;
                    tempFile << "\n" << customer.customer_record;
                }
                // tempFile << "\n" << newData.degree;
                
                
            }
        }
        read.close();
        tempFile.close();
        remove("customer.txt");
        rename("temp.txt", "customer.txt");
        cout << "\n\tData updated successfuly";
    }
    else {
        cout<<"\n\tNo change implemented";
        throw "\n\tRecord not updated";
    }
    }
}

void clearDueEmployee(){
    
    int id;
    try{
        id = searchDataEmployee();
    } catch(const char* msg){
                cerr<<msg<<endl;
                return;
    }
    
    if(checkUser(id)){cout << "\n\tYou want to clear due (y/n) : ";
    char choice;
    int option = 2;
    cin >> choice;
    while(std::cin.fail() || (choice!='Y' && choice!='y'&& choice!='n' && choice!='N')) {
    std::cin.clear();
    std::cin.ignore(std::numeric_limits<std::streamsize>::max(),'\n');
    std::cout << "\n\tInvaild input, Please enter (y/n) : ";
        cout<<"\n\t";
    std::cin >> choice;
    }

    if (choice == 'y' || choice =='Y') {
        Customer newData;

        
           if(option==2){
          
        newData.fineDue=0;
         while(std::cin.fail() || newData.fineDue<0) {
    std::cin.clear();
    std::cin.ignore(std::numeric_limits<std::streamsize>::max(),'\n');
    std::cout << "\n\tInvaild entry, Please enter (+)Number";
        cout<<"\n\t";
    std::cin >> newData.fineDue;
    }
        }

        Employee customer;
        ofstream tempFile;
        tempFile.open("temp.txt");
        ifstream read;
        read.open("employee.txt");
        while (!read.eof()) {
            read >> customer.id;
            read >> customer.password;

            read.ignore();
            getline(read, customer.rentedCars);
            getline(read, customer.name);

            read >> customer.fineDue;
            // read >> customer.degree;
            read >> customer.customer_record;
            if (customer.id != id) {
                tempFile << "\n" << customer.id;
                tempFile << "\n" << customer.password;
                tempFile << "\n" << customer.rentedCars;
        tempFile << "\n" << customer.name;
                tempFile << "\n" << customer.fineDue;
                // tempFile << "\n" << customer.degree;
                tempFile << "\n" << customer.customer_record;
            }
            else {
                if(option==2){
                    tempFile << "\n"<< customer.id;
                    tempFile << "\n" << customer.password;
                    tempFile << "\n" << customer.rentedCars;
                    tempFile << "\n" << customer.name;
                    tempFile << "\n"<< newData.fineDue;
                    tempFile << "\n" << customer.customer_record;
                }
                // tempFile << "\n" << newData.degree;
                
                
            }
        }
        read.close();
        tempFile.close();
        remove("employee.txt");
        rename("temp.txt", "employee.txt");
        cout << "\n\tData updated successfuly";
    }
    else {
        cout<<"\n\tNo change implemented";
        throw "\n\tRecord not updated";
    }
    }
}

int invalidInput(int i){
    if(i==-1){
        cout<<"\n\tInvalid input";
        cout<<"\n\tEnter correct input";
    }
    return 0;
}

int main()
{

    
     cout<<"\n\t\tWelcome User!";
    cout<<"\n\tLogin as : ";
    cout<<"\n\t\t1.Manager";
    cout<<"\n\t\t2.Customer";
    cout<<"\n\t\t3.Employee";
     cout << "\n\tEnter choice : ";

    int category;
    cin >> category;
        while(std::cin.fail() || category<1 || category>3) {
    std::cin.clear();
    std::cin.ignore(std::numeric_limits<std::streamsize>::max(),'\n');
    std::cout << "\n\tInvaild input\n";
      cout<<"\n\tLogin as : ";
    cout<<"\n\t\t1.Manager";
    cout<<"\n\t\t2.Customer";
    cout<<"\n\t\t3.Employee";
        cout << "\n\tEnter choice : ";
    std::cin >> category;
    }

     if(category==2){
        int id;
        int password;
        cout<<"\n\tNote: Enter 0 to Logout";
        cout<<"\n\tEnter your ID : ";
        cin>>id;
        while(std::cin.fail() || id<0 || ((id!=0)&&(!checkUser(id)))) {
    std::cin.clear();
    std::cin.ignore(std::numeric_limits<std::streamsize>::max(),'\n');
    if(cin.fail()){
        std::cout << "\n\tInvaild entry.  Enter a (+) DIGITS : ";
    }
    else if(!checkUser(id)){std::cout<<"\n\tNo such user ID exists \n\tEnter ID: ";}
    else{
        std::cout << "\n\tInvaild entry.  Enter a (+) DIGITS : ";
    }
    std::cin >> id;;
}
if(id==0){
    category=0;
}



if(category==2){

    cout<<"\n\tNote: Enter 0 to Logout";
        cout<<"\n\tEnter your Password : ";
        cin>>password;
        while(std::cin.fail() || password<0) {
    std::cin.clear();
    std::cin.ignore(std::numeric_limits<std::streamsize>::max(),'\n');
    // else if(!checkUser(id)){std::cout<<"\n\tNo such user ID exists \n\tEnter ID: ";}
        std::cout << "\n\tInvaild entry.  Enter a (+) DIGITS : ";
    std::cin >> password;
}
if(password==0){
    category=0;
}

} 
if(category==2){
    if(!validateCustomer(id,password)){
        cout<<"\n\tInvalid password";
    }
    else{
        Customer customer;
    customer.setCustomer(id);
        int option2;
        cout<<"\n\t--Welocome "<<customer.name<<"--";
        while (true)
        { 
        cout<<"\n\t1. Logout";
        cout<<"\n\t2. Profile";
        cout<<"\n\t3. Available cars";
        cout<<"\n\t4. Rentable cars";
        cout<<"\n\t5. Rent a car";
        cout<<"\n\t6. Rented car(s)";
        cout<<"\n\t7. Return car";
        cout << "\n\tEnter choice : ";
        cin>>option2;
        while(std::cin.fail() || option2<1 || option2>7) {
    std::cin.clear();
    std::cin.ignore(std::numeric_limits<std::streamsize>::max(),'\n');
    std::cout << "\n\tInvaild input\n";
     cout<<"\n\t1. Logout";
        cout<<"\n\t2. Profile";
        cout<<"\n\t3. Available cars";
        cout<<"\n\t4. Rentable cars";
        cout<<"\n\t5. Rent a car";
        cout<<"\n\t6. Rented car(s)";
        cout<<"\n\t7. Return car";
    cout << "\n\tEnter choice : ";
    std::cin >> option2;
    }
     
    
    if(option2==1){
    break;
    }
    if(option2==2){
        customer.printCustomer();
    }
    if(option2==3){
    readAvailableCarData();
    }
    if(option2==4){
    customer.readRentableCarData();
    }
    if(option2==5){
    // customer.rentaCar();
    try{
              customer.rentaCar();
            } catch(const char* msg){
                cerr<<msg<<endl;
            }
    }
    if(option2==6){
        cout<<"\n\tCurrently rented cars : \n";
        customer.rentedCarData();
    }
    if(option2==7){
        try{
              customer.returnRequest();
            } catch(const char* msg){
                cerr<<msg<<endl;
            }
    
    }
     }
}
    }
}


if(category==3){
        int id;
        int password;
        cout<<"\n\tNote: Enter 0 to Logout";
        cout<<"\n\tEnter your ID : ";
        cin>>id;
        while(std::cin.fail() || id<0 || ((id!=0)&&(!checkUser(id)))) {
    std::cin.clear();
    std::cin.ignore(std::numeric_limits<std::streamsize>::max(),'\n');
    if(cin.fail()){
        std::cout << "\n\tInvaild entry.  Enter a (+) DIGITS : ";
    }
    else if(!checkUser(id)){std::cout<<"\n\tNo such user ID exists \n\tEnter ID: ";}
    else{
        std::cout << "\n\tInvaild entry.  Enter a (+) DIGITS : ";
    }
    std::cin >> id;;
}
if(id==0){
    category=0;
}



if(category==3){

    cout<<"\n\tNote: Enter 0 to Logout";
        cout<<"\n\tEnter your Password : ";
        cin>>password;
        while(std::cin.fail() || password<0) {
    std::cin.clear();
    std::cin.ignore(std::numeric_limits<std::streamsize>::max(),'\n');
    // else if(!checkUser(id)){std::cout<<"\n\tNo such user ID exists \n\tEnter ID: ";}
        std::cout << "\n\tInvaild entry.  Enter a (+) DIGITS : ";
    std::cin >> password;
}
if(password==0){
    category=0;
}

} 
if(category==3){
    if(!validateEmployee(id,password)){
        cout<<"\n\tInvalid password";
    }
    else{
        Employee customer;
    customer.setEmployee(id);
        int option2;
        cout<<"\n\t--Welocome "<<customer.name<<"--";
        while (true)
        { 
        cout<<"\n\t1. Logout";
        cout<<"\n\t2. Profile";
        cout<<"\n\t3. Available cars";
        cout<<"\n\t4. Rentable cars";
        cout<<"\n\t5. Rent a car";
        cout<<"\n\t6. Rented car(s)";
        cout<<"\n\t7. Return car";
        cout << "\n\tEnter choice : ";
        cin>>option2;
        while(std::cin.fail() || option2<1 || option2>7) {
    std::cin.clear();
    std::cin.ignore(std::numeric_limits<std::streamsize>::max(),'\n');
    std::cout << "\n\tInvaild input\n";
     cout<<"\n\t1. Logout";
        cout<<"\n\t2. Profile";
        cout<<"\n\t3. Available cars";
        cout<<"\n\t4. Rentable cars";
        cout<<"\n\t5. Rent a car";
        cout<<"\n\t6. Rented car(s)";
        cout<<"\n\t7. Return car";
    cout << "\n\tEnter choice : ";
    std::cin >> option2;
    }
     
    
    if(option2==1){
    break;
    }
    if(option2==2){
        customer.printEmployee();
    }
    if(option2==3){
    readAvailableCarDataEmployee();
    }
    if(option2==4){
    customer.readRentableCarData();
    }
    if(option2==5){
    // customer.rentaCar();
    try{
              customer.rentaCar();
            } catch(const char* msg){
                cerr<<msg<<endl;
            }
    }
    if(option2==6){
        cout<<"\n\tCurrently rented cars : \n";
        customer.rentedCarData();
    }
    if(option2==7){
        try{
              customer.returnRequest();
            } catch(const char* msg){
                cerr<<msg<<endl;
            }
    
    }
     }
}
    }
}


  if(category==1){
        int ID;
        int Password;
        cout<<"\n\tNote: Enter 1 to Logout";
        cout<<"\n\tEnter ID : ";
        
        cin>>ID;
         while(std::cin.fail() || (ID!=0 && ID!=1)) {
    std::cin.clear();
    std::cin.ignore(std::numeric_limits<std::streamsize>::max(),'\n');
    std::cout << "\n\tInvaild ID\n";
    cout<<"\n\tNote: Enter 1 to Logout";
      cout<<"\n\tEnter ID : ";
      
    std::cin >> ID;
    if(ID==1){
        category=0;
        break;
    }
    }
    if (ID==1)
    {
        category=0;
    }
    
    if(ID==0){cout<<"\n\tNote: Enter 1 to Logout";
        cout<<"\n\tPassword : ";
        cin>>Password;
         while(std::cin.fail() || (Password!=0 && Password!=1)) {
    std::cin.clear();
    std::cin.ignore(std::numeric_limits<std::streamsize>::max(),'\n');
    std::cout << "\n\tInvaild Password\n";
    cout<<"\n\tNote: Enter 1 to Logout";
      cout<<"\n\tEnter Password : ";
    std::cin >> Password;
    if(Password==1){
        category=0;
        break;
    }
    }
    
    }
    if (Password==1)
    {
        category=0;
    }
    }


    if(category==1){
        cout<<"\n\tHello Manager!\n";
        // while (true)
        // {
    cout<<"\n\tManage : ";
    cout<<"\n\t1. Car Data";
    cout<<"\n\t2. Customer Data";
    cout<<"\n\t3. Employee Data";
    cout << "\n\tEnter choice : ";
    int option;
    cin>>option;
    while(std::cin.fail() || option<1 || option>3) {
    std::cin.clear();
    std::cin.ignore(std::numeric_limits<std::streamsize>::max(),'\n');
    std::cout << "\n\tInvaild input\n";
     cout<<"\n\tManage : ";
    cout<<"\n\t1. Car Data";
    cout<<"\n\t2. Customer Data";
    cout<<"\n\t3. Employee Data";
    cout << "\n\tEnter choice : ";
    std::cin >> option;
    }
    if(option==1){
        int choice;
    while (true)
    {
        cout<<"\n\t\t1. Add car data";
        cout<<"\n\t\t2. Read car data";
        cout<<"\n\t\t3. Search car data";
        cout<<"\n\t\t4. Delete car data";
        cout<<"\n\t\t5. Update car data";
        cout<<"\n\t\t6. Logout";
        cout << "\n\tEnter choice : ";
        cin>>choice;
        while(std::cin.fail() || choice<1 || choice>6) {
    std::cin.clear();
    std::cin.ignore(std::numeric_limits<std::streamsize>::max(),'\n');
    std::cout << "\n\tInvaild input\n";
      
    cout<<"\n\t\t1. Add car";
        cout<<"\n\t\t2. Read car";
        cout<<"\n\t\t3. Search car data";
        cout<<"\n\t\t4. Delete car data";
        cout<<"\n\t\t5. Update car data";
        cout<<"\n\t\t6. Logout";
        cout << "\n\tEnter choice : ";
    std::cin >> choice;
    }
        if(choice == 1){addCar();}
        else if(choice == 2){readCarData();}
        else if(choice == 3){
             try{
              searchCarData();
            } catch(const char* msg){
                cerr<<msg<<endl;
            }
        }
        else if(choice == 4){
             try{
              deleteCarData();
            } catch(const char* msg){
                cerr<<msg<<endl;
            }
        }
        else if(choice == 5){
             try{
              updateCarData();
            } catch(const char* msg){
                cerr<<msg<<endl;
            }
        }
        else if(choice==6){
            break;
        }
    }

    }
    else if(option==2 ){
        while (true) {
        cout << "\n\t1.Add customer data";
        cout << "\n\t2.See customer data";
        cout << "\n\t3.Search customer data";
        cout << "\n\t4.Delete customer data";
        cout << "\n\t5.Update customer data";
        cout << "\n\t6.Clear Due data";
                    cout << "\n\t7.Logout";

        int choice;
        cout << "\n\tEnter choice : ";
        cin >> choice;
        while(std::cin.fail() || choice<1 || choice>7) {
    std::cin.clear();
    std::cin.ignore(std::numeric_limits<std::streamsize>::max(),'\n');
    std::cout << "\n\tInvaild input\n";
      cout << "\n\t1.Add customer data";
        cout << "\n\t2.See customer data";
        cout << "\n\t3.Search customer data";
        cout << "\n\t4.Delete customer data";
        cout << "\n\t5.Update customer data";
        cout << "\n\t6.Clear Due";
        cout << "\n\t7.Logout";
        cout << "\n\tEnter choice : ";
    std::cin >> choice;
    }

        switch (choice) {
        case 1:
            addCustomer();
            try{
            
            } catch(const char* msg){
                cerr<<msg<<endl;
                
            }
            // addCustomer();
            // Customer::upCustomerCount();
            break;
        case 2:
            try{
               readDataCustomer();;
            } catch(const char* msg){
                cerr<<msg<<endl;
            }
            break;
        case 3:
            try{
               int id = searchDataCustomer();
            } catch(const char* msg){
                cerr<<msg<<endl;
            }
            // searchData();
            break;
        case 4:
        try{
              deleteCustomerData();
            } catch(const char* msg){
                cerr<<msg<<endl;
            }
            // deleteData();
            
            break;
        case 5:
            // updateData();
            try{
               updateDataCustomer();
            } catch(const char* msg){
                cerr<<msg<<endl;
                // checkUser(-1);
                invalidInput(-1);
            }
            
            break;
        case 6:
            // updateData();
            try{
               clearDueCustomer();
            } catch(const char* msg){
                cerr<<msg<<endl;
                // checkUser(-1);
                invalidInput(-1);
            }
            
            break;
        }
        if(choice==7){
            break;
        }
    }
    }
    else if(option==3 ){
        while (true) {
        cout << "\n\t1.Add employee data";
        cout << "\n\t2.See employee data";
        cout << "\n\t3.Search employee data";
        cout << "\n\t4.Delete employee data";
        cout << "\n\t5.Update employee data";
        cout << "\n\t6.Clear Due data";
                    cout << "\n\t7.Logout";
                    cout << "\n\tEnter choice : ";

        int choice;
        // cout << "\n\tEnter choice : ";
        cin >> choice;
        while(std::cin.fail() || choice<1 || choice>7) {
    std::cin.clear();
    std::cin.ignore(std::numeric_limits<std::streamsize>::max(),'\n');
    std::cout << "\n\tInvaild input\n";
      cout << "\n\t1.Add employee data";
        cout << "\n\t2.See employee data";
        cout << "\n\t3.Search employee data";
        cout << "\n\t4.Delete employee data";
        cout << "\n\t5.Update employee data";
        cout << "\n\t6.Clear Due data";
                    cout << "\n\t7.Logout";
                    cout << "\n\tEnter choice : ";

    std::cin >> choice;
    }

        switch (choice) {
        case 1:
            addEmployee();
            try{
            
            } catch(const char* msg){
                cerr<<msg<<endl;
                
            }
            // addCustomer();
            // Customer::upCustomerCount();
            break;
        case 2:
            try{
               readDataEmployee();;
            } catch(const char* msg){
                cerr<<msg<<endl;
            }
            break;
        case 3:
            try{
               int id = searchDataEmployee();
            } catch(const char* msg){
                cerr<<msg<<endl;
            }
            // searchData();
            break;
        case 4:
        try{
              deleteEmployeeData();
            } catch(const char* msg){
                cerr<<msg<<endl;
            }
            // deleteData();
            
            break;
        case 5:
            // updateData();
            try{
               updateDataEmployee();
            } catch(const char* msg){
                cerr<<msg<<endl;
                // checkUser(-1);
                invalidInput(-1);
            }
            
            break;
        case 6:
            // updateData();
            try{
               clearDueEmployee();
            } catch(const char* msg){
                cerr<<msg<<endl;
                // checkUser(-1);
                invalidInput(-1);
            }
            
            break;
        }
        if(choice==7){
            break;
        }
    }
    }

}
}

