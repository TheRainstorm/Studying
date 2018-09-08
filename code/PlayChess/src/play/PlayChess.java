package play;
import java.util.Scanner;
import javax.swing.JOptionPane;

public class PlayChess {
	private Map map=new Map();
	public static final int A=1;
	public static final int B=-1;
	
	public void play(int mode,int hard) {
		int result;
		print();
		while(true) {
			if(mode==0) {
				setPlayer();
			}
			else {
				setAI(hard);
			}
			print();
			
			result=check();
			if(result==9){
				System.out.println("It's a draw.");
			}
			else if(result==1){
				System.out.println("Congratulations, you win!");
				break;
			}
			else if(result==-1) {
				System.out.println("Game Over!");
				break;
			}
			
			if(mode==0) {
				setAI(hard);
			}
			else {
				setPlayer();
			}
			
			print();
			result=check();
			if(result==9){
				System.out.println("It's a draw.");
			}
			else if(result==1){
				System.out.println("Congratulations, you win!");
				break;
			}
			else if(result==-1) {
				System.out.println("Game Over!");
				break;
			}
		}
	}
	public void setPlayer() {
		System.out.println("Please enter the rows"
				+ " and cols separated by a space.");
//		Scanner scanner=new Scanner(System.in);
//		int row=Integer.parseInt(scanner.next());
//		int col=Integer.parseInt(scanner.next());
		int row,col;
		Scanner scanner;
		while(true) {
			String s=javax.swing.JOptionPane.showInputDialog(null);
			scanner=new Scanner(s);
			row=scanner.nextInt();
			col=scanner.nextInt();
			if(row<1||row>Map.MAX||col<1||col>Map.MAX) {
				continue;
			}
			if(map.get(row-1, col-1)==0) {
				map.set(row-1, col-1,A);
				scanner.close();
				break;
			}
		}
	}
	public void print(){
		map.print();
	}
	public int check() {
		return map.check();
	}
	public void setAI(int hard){
		if(hard==0) {
			while(true) {
				int row=(int)(Math.random()*Map.MAX);
				int col=(int)(Math.random()*Map.MAX);
				if(map.get(row, col)==0) {
					map.set(row, col, B);
					break;
				}
			}
		}
		else {
			
		}
	}

}
class Map{
	public static final int MAX=3;
	private int[][] map=new int[MAX][MAX];
	
	public void set(int i,int j,int flag) {
		map[i][j]=flag;
	}
	public int get(int i,int j) {
		return map[i][j];
	}
	public void print(){
		System.out.println("The Map:");
		for(int i=0;i<MAX;i++) {
			for(int j=0;j<MAX;j++) {
				if(map[i][j]==1) {
					System.out.print("A");
				}
				else if(map[i][j]==0) {
					System.out.print("*");
				}
				else {
					System.out.print("B");
				}
			}
			System.out.println();
		}
	}
	//结果 0:未分出胜负,1:A赢,-1:B赢。
	public int check() {
		int sum1,sum2,i,j;
		boolean noSpace=true;
		for(i=0;i<MAX;i++) {
			sum1=0;sum2=0;
			for(j=0;j<MAX;j++) {
				if(map[i][j]==0) {
					noSpace=false;
				}
				sum1+=map[i][j];
				sum2+=map[j][i];
			}
			if(sum1==MAX||sum2==MAX) {
				return 1;
			}
			else if(sum1==-1*MAX||sum2==-1*MAX) {
				return -1;
			}
		}
		
		sum1=sum2=0;
		for(i=0;i<MAX;i++) {
			sum1+=map[i][i];
			sum2+=map[i][MAX-i-1];
		}
		if(sum1==MAX||sum2==MAX) {
			return 1;
		}
		else if(sum1==-1*MAX||sum2==-1*MAX) {
			return -1;
		}
		
		if(noSpace) {
			return 9;
		}
		return 0;
	}
}