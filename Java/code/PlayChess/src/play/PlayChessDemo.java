package play;

public class PlayChessDemo {
	public static boolean yes(String str) {
		System.out.println(str+" y//n");
		String s=javax.swing.JOptionPane.showInputDialog(null);
		if(s.toLowerCase().equals("y")) {
			return true;
		}
		else {
			return false;
		}
	}

	public static void main(String[] args) {
		
		do {
			PlayChess p=new PlayChess();
			int mode,hard;
			if(yes("First hand?")) {
				mode=0;
			}
			else {
				mode=1;
			}
			if(yes("Difficult mode?")) {
				hard=1;
			}
			else{
				hard=0;
			}
			p.play(mode,hard);
		}while(yes("Do you want to play again?"));
		
	}

}
