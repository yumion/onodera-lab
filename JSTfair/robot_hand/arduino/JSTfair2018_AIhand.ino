#include <Servo.h>

Servo servo1;
//const uint8_t speeds = 100;

// グローバル変数の宣言
char input[1];  // 文字列格納用
int i = 0;      // 文字数のカウンタ
int val = 0;    // 受信した数値
int rot=0;   // 車輪回転と4本指開閉の指示用

void setup() {
  Serial.begin(9600);
  pinMode(14,OUTPUT); //PWM用のPINは全部使えない問題あり、アナログピンの逃がす。
  pinMode(16,OUTPUT);
  pinMode(17,OUTPUT);
  pinMode(18,OUTPUT);
  pinMode(19,OUTPUT);
  servo1.attach(14);
//  Serial.println("Input 2digits as CarMovement and HandGrip");
//  Serial.println("CarMovement: 53.=break, 49.=forward, 50.=back, 51.=cw, 52.=ccw");
//  Serial.println("HandGrip : 60=0degree, 61=30degree, 62=60degree, 63=90degree");
}


// シリアル通信で受信したデータを数値に変換
int serialNum(){
  // データ受信した場合の処理
  if (Serial.available()) {
    input[i] = Serial.read();
     // 文字数が3以上 or 末尾文字がある場合の処理
    if (i > 0 || input[i] == "\0") {
//    if (i > 2) {
      //input[i+1] = '\0';      // 末尾に終端文字の挿入
      val = atoi(input);    // 文字列を数値に変換
      Serial.write(input); // 文字列を送信
      Serial.write("\n");
      i = 0;      // カウンタの初期化
    }
    else { i++; }
  }
  return val;
}

int Wheel(){
    // put your main code here, to run repeatedly:
  // 正転(回転)
  if(rot == 49){
     //Serial.println("正転");
//     analogWrite(10,speeds);
//     analogWrite(11,0);
     digitalWrite(16,HIGH);
     digitalWrite(17,0);

//     analogWrite(5,speeds);
//     analogWrite(6,0);
     digitalWrite(18,HIGH);
     digitalWrite(19,0);

     //delay(5000);
  }
  else{
     // 逆転(逆回転)
     if(rot == 50){
        //Serial.println("逆転");
//        analogWrite(10,0);
//        analogWrite(11,speeds);
        digitalWrite(16,0);
        digitalWrite(17,HIGH);
        
//        analogWrite(5,0);
//        analogWrite(6,speeds);
        digitalWrite(18,0);
        digitalWrite(19,HIGH);
        //delay(5000);
     }
     else{
        if(rot == 51){
           //Serial.println("右周り");
//           analogWrite(10,0);
//           analogWrite(11,speeds);
           digitalWrite(16,0);
           digitalWrite(17,HIGH);

//           analogWrite(5,speeds);
//           analogWrite(6,0);      
           digitalWrite(18,HIGH);
           digitalWrite(19,0);

        }
        else{
          if(rot == 52){
            //Serial.println("左周り");
//           analogWrite(10,speeds);
//           analogWrite(11,0);
           digitalWrite(16,HIGH);
           digitalWrite(17,0);

//           analogWrite(5,0);
//           analogWrite(6,speeds);                 
           digitalWrite(18,0);
           digitalWrite(19,HIGH);
          }
          else{
           // ブレーキ
           //Serial.println("ブレーキ");
//           analogWrite(10,0);
//           analogWrite(11,0);
           digitalWrite(16,0);
           digitalWrite(17,0);

//           analogWrite(5,0);
//           analogWrite(6,0);
           digitalWrite(18,0);
           digitalWrite(19,0);
           //delay(1000);
          }
        }
     }
  }
}

int Hand(){
  if(rot == 60){
  servo1.write(0);
  }
  else{
    if(rot == 61){
    servo1.write(30);
    }
    else{
      if(rot == 62){
      servo1.write(60);
      }
      else{
        servo1.write(90);
      }
    }
  }
}


void loop() {
  rot = serialNum();

  if(rot >=45 && rot <= 59){
    Wheel();  
  }
  else{
    if(rot >= 60 && rot < 70){
      Hand();
    }
    else{
      servo1.write(90);
//      analogWrite(10,0);
//      analogWrite(11,0);
      digitalWrite(16,0);
      digitalWrite(17,0);

//      analogWrite(5,0);
//      analogWrite(6,0);      
        digitalWrite(18,0);
        digitalWrite(19,0);

    }
  }
 
}

