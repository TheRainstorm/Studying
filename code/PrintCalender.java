package time;
import javax.swing.JOptionPane;

public class PrintCalender {
	public static void print(int year,int month) {
		int totaldays=0;
		for(int i=1970;i<year;i++) {
			totaldays+=getTheDaysOfTheYear(i);
		}
		for(int i=1;i<month;i++){
			totaldays+=getTheDaysOfTheMonth(year,i);
		}
		int dayOfWeek=(4+totaldays)%7;
		int daysOfTheMonth=getTheDaysOfTheMonth(year,month);
		
		System.out.println(" Sun Mon Tue Wed Thu Fri Sat");
		for(int i=0;i<dayOfWeek+daysOfTheMonth;i++) {
			if(i<dayOfWeek) {
				System.out.printf("%4s"," ");
			}
			else {
				System.out.printf("%4s",String.valueOf(i-dayOfWeek+1));
			}
			if(i%7==6) {
				System.out.println();
			}
		}
	}
	public static boolean isLeapYear(int year) {
		if((year%4==0&&year%100!=0)||year%400==0) {
			return true;
		}
		else {
			return false;
		}
	}
	public static int getTheDaysOfTheYear(int year) {
		if(isLeapYear(year)) {
			return 366;
		}
		else {
			return 365;
		}
	}
	public static int getTheDaysOfTheMonth(int year,int month) {
		if(month==1||month==3||month==5||month==7||month==8||
				month==10||month==12) {
			return 31;
		}
		else if(month==2) {
			if(isLeapYear(year)) {
				return 29;
			}
			else {
				return 28;
			}
		}
		else {
			return 30;
		}
	}
	public static void main(String[] args) {
		String year=JOptionPane.showInputDialog("Enter the full year");
		String month=JOptionPane.showInputDialog("Enter the month");
		
		print(Integer.parseInt(year),Integer.parseInt(month));

	}

}
