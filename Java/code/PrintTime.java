package time;

public class PrintTime {
	public static void print(long mseconds) {
		long seconds=mseconds/1000;
		long minutes=seconds/60;
		int secondsLeft=(int)(seconds%60);
		long hours=minutes/60;
		int minutesLeft=(int)(minutes%60);
		int hoursLeft=(int)(hours%24);
		long days= hours/24;
		
		int dayOfWeekInNumber=(int)((4+days)%7);
		String dayOfWeek=getDayOfWeek(dayOfWeekInNumber);
		//输出时间
		System.out.println(hoursLeft+":"+minutesLeft+":"+secondsLeft+" GMT"
				+" "+dayOfWeek);
		//获得年份
		int i;
		for(i=1970;;i++) {
			int temp=getTheDaysOfTheYear(i);
			if(days<temp) {
				break;
			}
			else {
				days-=temp;
			}
		}
		int year=i;
		//获得月份
		int j;
		for(j=1;;j++) {
			int temp=getTheDaysOfTheMonth(year,j);
			if(days<temp) {
				break;
			}
			else {
				days-=temp;
			}
		}
		int month=j;
		int daysLeft=(int)days+1;
		
		//输出日期
		System.out.printf("%d//%d//%d",year,month,daysLeft);
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
	public static String getDayOfWeek(int num) {
		switch(num) {
			case 1:	return "Monday";
			case 2: return "Tuesday";
			case 3: return "Wednesday";
			case 4: return "Thursday";
			case 5: return "Friday";
			case 6: return "Saturday";
			case 0: return "Sunday";
			default: return "YouCanNeverSeeThis";
		}
	}

	public static void main(String[] args) {
		long mseconds=System.currentTimeMillis();
		print(mseconds);
	}

}
