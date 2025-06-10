<?xml version="1.0" encoding="utf-8"?>
<androidx.constraintlayout.widget.ConstraintLayout
    xmlns:android="http://schemas.android.com/apk/res/android"
    xmlns:app="http://schemas.android.com/apk/res-auto"
    xmlns:tools="http://schemas.android.com/tools"
    android:layout_width="match_parent"
    android:layout_height="match_parent"
    android:background="#F0F0F0"
    tools:context=".MainActivity">

    <!-- Calculator Display -->
    <TextView
        android:id="@+id/displayText"
        android:layout_width="0dp"
        android:layout_height="0dp"
        android:layout_marginStart="16dp"
        android:layout_marginTop="32dp"
        android:layout_marginEnd="16dp"
        android:gravity="end|bottom"
        android:padding="24dp"
        android:text="0"
        android:textColor="#000000"
        android:textSize="48sp"
        android:textStyle="bold"
        app:layout_constraintBottom_toTopOf="@+id/guideline"
        app:layout_constraintEnd_toEndOf="parent"
        app:layout_constraintStart_toStartOf="parent"
        app:layout_constraintTop_toTopOf="parent" />

    <androidx.constraintlayout.widget.Guideline
        android:id="@+id/guideline"
        android:layout_width="wrap_content"
        android:layout_height="wrap_content"
        android:orientation="horizontal"
        app:layout_constraintGuide_percent="0.3" />

    <androidx.gridlayout.widget.GridLayout
        android:layout_width="0dp"
        android:layout_height="0dp"
        android:layout_marginStart="16dp"
        android:layout_marginTop="16dp"
        android:layout_marginEnd="16dp"
        android:layout_marginBottom="16dp"
        app:columnCount="4"
        app:layout_constraintBottom_toBottomOf="parent"
        app:layout_constraintEnd_toEndOf="parent"
        app:layout_constraintStart_toStartOf="parent"
        app:layout_constraintTop_toBottomOf="@+id/displayText"
        app:rowCount="5">

        <!-- Row 1: AC, +/-, %, ÷ -->
        <Button
            android:id="@+id/buttonClear"
            android:layout_width="0dp"
            android:layout_height="0dp"
            android:layout_margin="4dp"
            android:text="AC"
            android:textSize="24sp"
            android:textStyle="bold"
            android:backgroundTint="#A5A5A5"
            android:textColor="#212121"
            app:cornerRadius="24dp"
            android:elevation="4dp"
            app:layout_column="0"
            app:layout_columnWeight="1"
            app:layout_row="0"
            app:layout_rowWeight="1" />

        <Button
            android:id="@+id/buttonPlusMinus"
            android:layout_width="0dp"
            android:layout_height="0dp"
            android:layout_margin="4dp"
            android:text="±"
            android:textSize="24sp"
            android:textStyle="bold"
            android:backgroundTint="#A5A5A5"
            android:textColor="#212121"
            app:cornerRadius="24dp"
            android:elevation="4dp"
            app:layout_column="1"
            app:layout_columnWeight="1"
            app:layout_row="0"
            app:layout_rowWeight="1" />

        <Button
            android:id="@+id/buttonPercent"
            android:layout_width="0dp"
            android:layout_height="0dp"
            android:layout_margin="4dp"
            android:text="%"
            android:textSize="24sp"
            android:textStyle="bold"
            android:backgroundTint="#A5A5A5"
            android:textColor="#212121"
            app:cornerRadius="24dp"
            android:elevation="4dp"
            app:layout_column="2"
            app:layout_columnWeight="1"
            app:layout_row="0"
            app:layout_rowWeight="1" />

        <Button
            android:id="@+id/buttonDivide"
            android:layout_width="0dp"
            android:layout_height="0dp"
            android:layout_margin="4dp"
            android:text="÷"
            android:textSize="24sp"
            android:textStyle="bold"
            android:backgroundTint="#FF9500"
            android:textColor="#FFFFFF"
            app:cornerRadius="24dp"
            android:elevation="4dp"
            app:layout_column="3"
            app:layout_columnWeight="1"
            app:layout_row="0"
            app:layout_rowWeight="1" />

        <!-- Row 2: 7, 8, 9, × -->
        <Button
            android:id="@+id/button7"
            android:layout_width="0dp"
            android:layout_height="0dp"
            android:layout_margin="4dp"
            android:text="7"
            android:textSize="24sp"
            android:textStyle="bold"
            android:backgroundTint="#FFFFFF"
            android:textColor="#212121"
            app:cornerRadius="24dp"
            android:elevation="4dp"
            app:layout_column="0"
            app:layout_columnWeight="1"
            app:layout_row="1"
            app:layout_rowWeight="1" />

        <Button
            android:id="@+id/button8"
            android:layout_width="0dp"
            android:layout_height="0dp"
            android:layout_margin="4dp"
            android:text="8"
            android:textSize="24sp"
            android:textStyle="bold"
            android:backgroundTint="#FFFFFF"
            android:textColor="#212121"
            app:cornerRadius="24dp"
            android:elevation="4dp"
            app:layout_column="1"
            app:layout_columnWeight="1"
            app:layout_row="1"
            app:layout_rowWeight="1" />

        <Button
            android:id="@+id/button9"
            android:layout_width="0dp"
            android:layout_height="0dp"
            android:layout_margin="4dp"
            android:text="9"
            android:textSize="24sp"
            android:textStyle="bold"
            android:backgroundTint="#FFFFFF"
            android:textColor="#212121"
            app:cornerRadius="24dp"
            android:elevation="4dp"
            app:layout_column="2"
            app:layout_columnWeight="1"
            app:layout_row="1"
            app:layout_rowWeight="1" />

        <Button
            android:id="@+id/buttonMultiply"
            android:layout_width="0dp"
            android:layout_height="0dp"
            android:layout_margin="4dp"
            android:text="×"
            android:textSize="24sp"
            android:textStyle="bold"
            android:backgroundTint="#FF9500"
            android:textColor="#FFFFFF"
            app:cornerRadius="24dp"
            android:elevation="4dp"
            app:layout_column="3"
            app:layout_columnWeight="1"
            app:layout_row="1"
            app:layout_rowWeight="1" />

        <!-- Row 3: 4, 5, 6, - -->
        <Button
            android:id="@+id/button4"
            android:layout_width="0dp"
            android:layout_height="0dp"
            android:layout_margin="4dp"
            android:text="4"
            android:textSize="24sp"
            android:textStyle="bold"
            android:backgroundTint="#FFFFFF"
            android:textColor="#212121"
            app:cornerRadius="24dp"
            android:elevation="4dp"
            app:layout_column="0"
            app:layout_columnWeight="1"
            app:layout_row="2"
            app:layout_rowWeight="1" />

        <Button
            android:id="@+id/button5"
            android:layout_width="0dp"
            android:layout_height="0dp"
            android:layout_margin="4dp"
            android:text="5"
            android:textSize="24sp"
            android:textStyle="bold"
            android:backgroundTint="#FFFFFF"
            android:textColor="#212121"
            app:cornerRadius="24dp"
            android:elevation="4dp"
            app:layout_column="1"
            app:layout_columnWeight="1"
            app:layout_row="2"
            app:layout_rowWeight="1" />

        <Button
            android:id="@+id/button6"
            android:layout_width="0dp"
            android:layout_height="0dp"
            android:layout_margin="4dp"
            android:text="6"
            android:textSize="24sp"
            android:textStyle="bold"
            android:backgroundTint="#FFFFFF"
            android:textColor="#212121"
            app:cornerRadius="24dp"
            android:elevation="4dp"
            app:layout_column="2"
            app:layout_columnWeight="1"
            app:layout_row="2"
            app:layout_rowWeight="1" />

        <Button
            android:id="@+id/buttonSubtract"
            android:layout_width="0dp"
            android:layout_height="0dp"
            android:layout_margin="4dp"
            android:text="-"
            android:textSize="24sp"
            android:textStyle="bold"
            android:backgroundTint="#FF9500"
            android:textColor="#FFFFFF"
            app:cornerRadius="24dp"
            android:elevation="4dp"
            app:layout_column="3"
            app:layout_columnWeight="1"
            app:layout_row="2"
            app:layout_rowWeight="1" />

        <!-- Row 4: 1, 2, 3, + -->
        <Button
            android:id="@+id/button1"
            android:layout_width="0dp"
            android:layout_height="0dp"
            android:layout_margin="4dp"
            android:text="1"
            android:textSize="24sp"
            android:textStyle="bold"
            android:backgroundTint="#FFFFFF"
            android:textColor="#212121"
            app:cornerRadius="24dp"
            android:elevation="4dp"
            app:layout_column="0"
            app:layout_columnWeight="1"
            app:layout_row="3"
            app:layout_rowWeight="1" />

        <Button
            android:id="@+id/button2"
            android:layout_width="0dp"
            android:layout_height="0dp"
            android:layout_margin="4dp"
            android:text="2"
            android:textSize="24sp"
            android:textStyle="bold"
            android:backgroundTint="#FFFFFF"
            android:textColor="#212121"
            app:cornerRadius="24dp"
            android:elevation="4dp"
            app:layout_column="1"
            app:layout_columnWeight="1"
            app:layout_row="3"
            app:layout_rowWeight="1" />

        <Button
            android:id="@+id/button3"
            android:layout_width="0dp"
            android:layout_height="0dp"
            android:layout_margin="4dp"
            android:text="3"
            android:textSize="24sp"
            android:textStyle="bold"
            android:backgroundTint="#FFFFFF"
            android:textColor="#212121"
            app:cornerRadius="24dp"
            android:elevation="4dp"
            app:layout_column="2"
            app:layout_columnWeight="1"
            app:layout_row="3"
            app:layout_rowWeight="1" />

        <Button
            android:id="@+id/buttonAdd"
            android:layout_width="0dp"
            android:layout_height="0dp"
            android:layout_margin="4dp"
            android:text="+"
            android:textSize="24sp"
            android:textStyle="bold"
            android:backgroundTint="#FF9500"
            android:textColor="#FFFFFF"
            app:cornerRadius="24dp"
            android:elevation="4dp"
            app:layout_column="3"
            app:layout_columnWeight="1"
            app:layout_row="3"
            app:layout_rowWeight="1" />

        <!-- Row 5: 0, ., ⌫, = -->
        <Button
            android:id="@+id/button0"
            android:layout_width="0dp"
            android:layout_height="0dp"
            android:layout_margin="4dp"
            android:text="0"
            android:textSize="24sp"
            android:textStyle="bold"
            android:backgroundTint="#FFFFFF"
            android:textColor="#212121"
            app:cornerRadius="24dp"
            android:elevation="4dp"
            app:layout_column="0"
            app:layout_columnWeight="1"
            app:layout_row="4"
            app:layout_rowWeight="1" />

        <Button
            android:id="@+id/buttonDecimal"
            android:layout_width="0dp"
            android:layout_height="0dp"
            android:layout_margin="4dp"
            android:text="."
            android:textSize="24sp"
            android:textStyle="bold"
            android:backgroundTint="#FFFFFF"
            android:textColor="#212121"
            app:cornerRadius="24dp"
            android:elevation="4dp"
            app:layout_column="1"
            app:layout_columnWeight="1"
            app:layout_row="4"
            app:layout_rowWeight="1" />

        <Button
            android:id="@+id/buttonBackspace"
            android:layout_width="0dp"
            android:layout_height="0dp"
            android:layout_margin="4dp"
            android:text="⌫"
            android:textSize="24sp"
            android:textStyle="bold"
            android:backgroundTint="#FFFFFF"
            android:textColor="#212121"
            app:cornerRadius="24dp"
            android:elevation="4dp"
            app:layout_column="2"
            app:layout_columnWeight="1"
            app:layout_row="4"
            app:layout_rowWeight="1" />

        <Button
            android:id="@+id/buttonEquals"
            android:layout_width="0dp"
            android:layout_height="0dp"
            android:layout_margin="4dp"
            android:text="="
            android:textSize="24sp"
            android:textStyle="bold"
            android:backgroundTint="#FF9500"
            android:textColor="#FFFFFF"
            app:cornerRadius="24dp"
            android:elevation="4dp"
            app:layout_column="3"
            app:layout_columnWeight="1"
            app:layout_row="4"
            app:layout_rowWeight="1" />

    </androidx.gridlayout.widget.GridLayout>

    <!-- Display background drawable defined inline -->


</androidx.constraintlayout.widget.ConstraintLayout>